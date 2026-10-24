import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "all_cleaned.parquet"
MAP_FILE = BASE_DIR / "data" / "processed" / "label_map_model.json"
OUT_DIR = BASE_DIR / "outputs"

N_FOLDS = 5
GAP_FRAC = 0.02          # rows near the test block that are removed from training
BENIGN_CAP = 300_000
ATTACK_CAP = 100_000

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)

# ---------------------------------------------------------
# 1. Class map (same as stage5_cv.py)
# ---------------------------------------------------------
with open(MAP_FILE) as f:
    raw_map = json.load(f)
if all(isinstance(v, int) for v in raw_map.values()):
    label_to_id = dict(raw_map)
else:
    label_to_id = {name: int(i) for i, name in raw_map.items()}
id_to_label = {i: n for n, i in label_to_id.items()}
class_ids = sorted(id_to_label)
class_names = [id_to_label[i] for i in class_ids]
benign_id = label_to_id["BENIGN"]

# ---------------------------------------------------------
# 2. Load data (row order = capture order, we do NOT shuffle)
# ---------------------------------------------------------
df = pd.read_parquet(DATA_FILE)
labels = df["Label"].astype(str).str.strip()
labels = labels.where(~labels.str.startswith("Web Attack"), "Web Attack")
keep = labels.isin(label_to_id)

feature_cols = [c for c in df.columns if c not in ("Label", "Target")]
X = df.loc[keep, feature_cols].to_numpy(dtype=np.float32)
y = labels[keep].map(label_to_id).to_numpy(dtype=np.int8)
del df, labels, keep
print("Rows used:", len(y), "| Features:", X.shape[1])

# ---------------------------------------------------------
# 3. Assign each row to a block, in capture order, within its class
# ---------------------------------------------------------
block = np.zeros(len(y), dtype=np.int8)      # which of the 5 blocks
rank = np.zeros(len(y), dtype=np.int64)      # position within its own class
class_size = {}
for c in class_ids:
    idx = np.flatnonzero(y == c)             # ascending = capture order
    r = np.arange(len(idx))
    rank[idx] = r
    block[idx] = r * N_FOLDS // len(idx)
    class_size[c] = len(idx)

# ---------------------------------------------------------
# 4. Blocked cross-validation
# ---------------------------------------------------------
overall_rows = []
recall_rows, precision_rows, f1_rows = [], [], []
total_start = time.time()

for fold in range(N_FOLDS):
    fold_start = time.time()
    rng = np.random.default_rng(100 + fold)
    test_idx = np.flatnonzero(block == fold)

    train_parts = []
    gap_removed = 0
    for c in class_ids:
        in_c = (y == c)
        r_test = rank[in_c & (block == fold)]
        lo, hi = r_test.min(), r_test.max()
        gap = max(1, int(GAP_FRAC * class_size[c]))
        usable = in_c & ((rank < lo - gap) | (rank > hi + gap))
        gap_removed += int((in_c & (block != fold)).sum() - usable.sum())

        idx_c = np.flatnonzero(usable)
        cap = BENIGN_CAP if c == benign_id else ATTACK_CAP
        if len(idx_c) > cap:
            idx_c = rng.choice(idx_c, size=cap, replace=False)
        train_parts.append(idx_c)
    train_idx = np.concatenate(train_parts)

    model = RandomForestClassifier(
        n_estimators=100, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42,
    )
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    y_te = y[test_idx]

    acc = accuracy_score(y_te, pred)
    macro_f1 = f1_score(y_te, pred, average="macro", labels=class_ids, zero_division=0)
    true_attack = y_te != benign_id
    pred_attack = pred != benign_id
    attack_caught = (true_attack & pred_attack).sum() / true_attack.sum()
    false_alarm = (~true_attack & pred_attack).sum() / (~true_attack).sum()

    overall_rows.append({
        "fold": fold + 1, "train_rows": len(train_idx), "test_rows": len(y_te),
        "gap_rows_removed": gap_removed,
        "accuracy": acc, "macro_f1": macro_f1,
        "attacks_caught": attack_caught, "false_alarm_rate": false_alarm,
    })
    recall_rows.append(recall_score(y_te, pred, average=None, labels=class_ids, zero_division=0))
    precision_rows.append(precision_score(y_te, pred, average=None, labels=class_ids, zero_division=0))
    f1_rows.append(f1_score(y_te, pred, average=None, labels=class_ids, zero_division=0))

    print(f"Fold {fold + 1}/{N_FOLDS}: accuracy={acc:.4f}  macro-F1={macro_f1:.4f}  "
          f"attacks caught={attack_caught:.4%}  false alarms={false_alarm:.4%}  "
          f"(gap removed {gap_removed:,} rows, {time.time() - fold_start:.0f}s)")

# ---------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------
overall = pd.DataFrame(overall_rows)
metric_cols = ["accuracy", "macro_f1", "attacks_caught", "false_alarm_rate"]

print("\n" + "=" * 70)
print("BLOCKED CV: OVERALL (mean and spread across folds)")
print("=" * 70)
print(pd.DataFrame({"mean": overall[metric_cols].mean(),
                    "std": overall[metric_cols].std(),
                    "min": overall[metric_cols].min(),
                    "max": overall[metric_cols].max()}).round(5))

recall_arr, precision_arr, f1_arr = map(np.array, (recall_rows, precision_rows, f1_rows))
per_class = pd.DataFrame({
    "recall_mean": recall_arr.mean(axis=0), "recall_std": recall_arr.std(axis=0, ddof=1),
    "precision_mean": precision_arr.mean(axis=0), "precision_std": precision_arr.std(axis=0, ddof=1),
    "f1_mean": f1_arr.mean(axis=0), "f1_std": f1_arr.std(axis=0, ddof=1),
}, index=class_names)

print("\n" + "=" * 70)
print("BLOCKED CV: PER CLASS (mean and spread across folds)")
print("=" * 70)
print(per_class.round(4))

# --- Side-by-side with the random-fold results from A1 ---
try:
    a1_overall = pd.read_csv(OUT_DIR / "cv_results_overall.csv")
    a1_per = pd.read_csv(OUT_DIR / "cv_results_per_class.csv", index_col=0)

    print("\n" + "=" * 70)
    print("RANDOM CV (A1) vs BLOCKED CV (A2b): OVERALL MEANS")
    print("=" * 70)
    print(pd.DataFrame({
        "random_cv": a1_overall[metric_cols].mean(),
        "blocked_cv": overall[metric_cols].mean(),
    }).round(5))

    print("\n" + "=" * 70)
    print("RANDOM CV (A1) vs BLOCKED CV (A2b): PER CLASS")
    print("=" * 70)
    comparison = pd.DataFrame({
        "recall_random": a1_per["recall_mean"], "recall_blocked": per_class["recall_mean"],
        "precision_random": a1_per["precision_mean"], "precision_blocked": per_class["precision_mean"],
        "f1_random": a1_per["f1_mean"], "f1_blocked": per_class["f1_mean"],
    })
    comparison["f1_change"] = comparison["f1_blocked"] - comparison["f1_random"]
    print(comparison.round(4))
except FileNotFoundError:
    print("\n(A1 result files not found, skipping the comparison)")

overall.to_csv(OUT_DIR / "blocked_cv_results_overall.csv", index=False)
per_class.to_csv(OUT_DIR / "blocked_cv_results_per_class.csv")
print("\nSaved: outputs/blocked_cv_results_overall.csv and outputs/blocked_cv_results_per_class.csv")
print(f"Total time: {(time.time() - total_start) / 60:.1f} minutes")