import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "all_cleaned.parquet"
MAP_FILE = BASE_DIR / "data" / "processed" / "label_map_model.json"
OUT_DIR = BASE_DIR / "outputs"

N_FOLDS = 5
BENIGN_CAP = 300_000
ATTACK_CAP = 100_000

# ---------------------------------------------------------
# 1. Load the class map (accepts name->id or id->name layouts)
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
print("Classes:", dict(zip(class_ids, class_names)))

# ---------------------------------------------------------
# 2. Load the data and rebuild the 11 classes
# ---------------------------------------------------------
df = pd.read_parquet(DATA_FILE)
labels = df["Label"].astype(str).str.strip()
labels = labels.where(~labels.str.startswith("Web Attack"), "Web Attack")

keep = labels.isin(label_to_id)
print("\nRows loaded:", len(df))
print("Rows dropped (classes not in the 11-class map):", int((~keep).sum()))
print("Dropped labels:", labels[~keep].value_counts().to_dict())

feature_cols = [c for c in df.columns if c not in ("Label", "Target")]
X = df.loc[keep, feature_cols].to_numpy(dtype=np.float32)
y = labels[keep].map(label_to_id).to_numpy(dtype=np.int8)
del df, labels, keep   # free memory

print("Rows used:", len(y), "| Features:", X.shape[1])

# ---------------------------------------------------------
# 3. Cross-validation
# ---------------------------------------------------------
def cap_training_indices(train_idx, rng):
    """Same capping idea as Stage 4: limit BENIGN and each attack class."""
    parts = []
    for c in np.unique(y[train_idx]):
        idx_c = train_idx[y[train_idx] == c]
        cap = BENIGN_CAP if c == benign_id else ATTACK_CAP
        if len(idx_c) > cap:
            idx_c = rng.choice(idx_c, size=cap, replace=False)
        parts.append(idx_c)
    return np.concatenate(parts)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

overall_rows = []
recall_rows, precision_rows, f1_rows = [], [], []
total_start = time.time()

for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
    fold_start = time.time()
    rng = np.random.default_rng(42 + fold)
    train_idx = cap_training_indices(train_idx, rng)

    model = RandomForestClassifier(
        n_estimators=100, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42,
    )
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    y_te = y[test_idx]

    # Overall metrics
    acc = accuracy_score(y_te, pred)
    macro_f1 = f1_score(y_te, pred, average="macro", labels=class_ids, zero_division=0)

    # Binary view: BENIGN vs any attack
    true_attack = y_te != benign_id
    pred_attack = pred != benign_id
    attack_caught = (true_attack & pred_attack).sum() / true_attack.sum()
    false_alarm = (~true_attack & pred_attack).sum() / (~true_attack).sum()

    overall_rows.append({
        "fold": fold, "train_rows": len(train_idx), "test_rows": len(y_te),
        "accuracy": acc, "macro_f1": macro_f1,
        "attacks_caught": attack_caught, "false_alarm_rate": false_alarm,
    })
    recall_rows.append(recall_score(y_te, pred, average=None, labels=class_ids, zero_division=0))
    precision_rows.append(precision_score(y_te, pred, average=None, labels=class_ids, zero_division=0))
    f1_rows.append(f1_score(y_te, pred, average=None, labels=class_ids, zero_division=0))

    print(f"Fold {fold}/{N_FOLDS}: accuracy={acc:.4f}  macro-F1={macro_f1:.4f}  "
          f"attacks caught={attack_caught:.4%}  false alarms={false_alarm:.4%}  "
          f"({time.time() - fold_start:.0f}s)")

# ---------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------
pd.set_option("display.width", 200)
overall = pd.DataFrame(overall_rows)
metric_cols = ["accuracy", "macro_f1", "attacks_caught", "false_alarm_rate"]

print("\n" + "=" * 60)
print("OVERALL (mean and spread across folds)")
print("=" * 60)
summary = pd.DataFrame({"mean": overall[metric_cols].mean(),
                        "std": overall[metric_cols].std(),
                        "min": overall[metric_cols].min(),
                        "max": overall[metric_cols].max()})
print(summary.round(5))

recall_arr, precision_arr, f1_arr = map(np.array, (recall_rows, precision_rows, f1_rows))
per_class = pd.DataFrame({
    "recall_mean": recall_arr.mean(axis=0), "recall_std": recall_arr.std(axis=0, ddof=1),
    "precision_mean": precision_arr.mean(axis=0), "precision_std": precision_arr.std(axis=0, ddof=1),
    "f1_mean": f1_arr.mean(axis=0), "f1_std": f1_arr.std(axis=0, ddof=1),
}, index=class_names)

print("\n" + "=" * 60)
print("PER CLASS (mean and spread across folds)")
print("=" * 60)
print(per_class.round(4))

overall.to_csv(OUT_DIR / "cv_results_overall.csv", index=False)
per_class.to_csv(OUT_DIR / "cv_results_per_class.csv")
print("\nSaved: outputs/cv_results_overall.csv and outputs/cv_results_per_class.csv")
print(f"Total time: {(time.time() - total_start) / 60:.1f} minutes")