import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "all_cleaned.parquet"
MAP_FILE = BASE_DIR / "data" / "processed" / "label_map_model.json"
OUT_DIR = BASE_DIR / "outputs"

N_FOLDS = 5
GAP_FRAC = 0.02
ATTACK_CAP = 100_000

CONFIGS = [
    {"name": "baseline",        "benign_cap": 300_000, "min_samples_leaf": 1, "drop": []},
    {"name": "benign_cap_600k", "benign_cap": 600_000, "min_samples_leaf": 1, "drop": []},
    {"name": "min_leaf_5",      "benign_cap": 300_000, "min_samples_leaf": 5, "drop": []},
    {"name": "no_dest_port",    "benign_cap": 300_000, "min_samples_leaf": 1, "drop": ["Destination Port"]},
]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)

# --- Class map ---
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
pos = {n: class_ids.index(label_to_id[n]) for n in ("Bot", "DoS slowloris", "DoS Slowhttptest")}

# --- Data (row order = capture order) ---
df = pd.read_parquet(DATA_FILE)
labels = df["Label"].astype(str).str.strip()
labels = labels.where(~labels.str.startswith("Web Attack"), "Web Attack")
keep = labels.isin(label_to_id)
feature_cols = [c for c in df.columns if c not in ("Label", "Target")]
X = df.loc[keep, feature_cols].to_numpy(dtype=np.float32)
y = labels[keep].map(label_to_id).to_numpy(dtype=np.int8)
del df, labels, keep
print("Rows used:", len(y), "| Features:", len(feature_cols))

# --- Blocks within each class, in capture order ---
block = np.zeros(len(y), dtype=np.int8)
rank = np.zeros(len(y), dtype=np.int64)
class_size = {}
for c in class_ids:
    idx = np.flatnonzero(y == c)
    r = np.arange(len(idx))
    rank[idx] = r
    block[idx] = r * N_FOLDS // len(idx)
    class_size[c] = len(idx)


def run_config(cfg):
    keep_cols = [i for i, c in enumerate(feature_cols) if c not in cfg["drop"]]
    folds = []
    for fold in range(N_FOLDS):
        start = time.time()
        rng = np.random.default_rng(100 + fold)
        test_idx = np.flatnonzero(block == fold)

        train_parts = []
        for c in class_ids:
            in_c = (y == c)
            r_test = rank[in_c & (block == fold)]
            lo, hi = r_test.min(), r_test.max()
            gap = max(1, int(GAP_FRAC * class_size[c]))
            idx_c = np.flatnonzero(in_c & ((rank < lo - gap) | (rank > hi + gap)))
            cap = cfg["benign_cap"] if c == benign_id else ATTACK_CAP
            if len(idx_c) > cap:
                idx_c = rng.choice(idx_c, size=cap, replace=False)
            train_parts.append(idx_c)
        train_idx = np.concatenate(train_parts)

        model = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, class_weight="balanced_subsample",
            min_samples_leaf=cfg["min_samples_leaf"], random_state=42,
        )
        model.fit(X[train_idx][:, keep_cols], y[train_idx])
        pred = model.predict(X[test_idx][:, keep_cols])
        y_te = y[test_idx]

        true_attack = y_te != benign_id
        pred_attack = pred != benign_id
        folds.append({
            "macro_f1": f1_score(y_te, pred, average="macro", labels=class_ids, zero_division=0),
            "false_alarm": (~true_attack & pred_attack).sum() / (~true_attack).sum(),
            "attacks_caught": (true_attack & pred_attack).sum() / true_attack.sum(),
            "recall": recall_score(y_te, pred, average=None, labels=class_ids, zero_division=0),
            "precision": precision_score(y_te, pred, average=None, labels=class_ids, zero_division=0),
        })
        print(f"  [{cfg['name']}] fold {fold + 1}/{N_FOLDS} done ({time.time() - start:.0f}s)")
    return folds


summary = {}
total_start = time.time()
for cfg in CONFIGS:
    print(f"\nRunning config: {cfg['name']}")
    folds = run_config(cfg)
    f1s = np.array([f["macro_f1"] for f in folds])
    fas = np.array([f["false_alarm"] for f in folds])
    recall = np.array([f["recall"] for f in folds])
    precision = np.array([f["precision"] for f in folds])
    summary[cfg["name"]] = {
        "macro_f1_mean": f1s.mean(),
        "macro_f1_worst_fold": f1s.min(),
        "macro_f1_std": f1s.std(ddof=1),
        "false_alarm_mean": fas.mean(),
        "false_alarm_worst_fold": fas.max(),
        "attacks_caught_mean": np.mean([f["attacks_caught"] for f in folds]),
        "Bot_recall_mean": recall[:, pos["Bot"]].mean(),
        "Bot_precision_mean": precision[:, pos["Bot"]].mean(),
        "slowloris_recall_mean": recall[:, pos["DoS slowloris"]].mean(),
        "Slowhttptest_recall_mean": recall[:, pos["DoS Slowhttptest"]].mean(),
        "fold_macro_f1": " ".join(f"{v:.3f}" for v in f1s),
    }
    print(f"  -> macro-F1 mean {f1s.mean():.4f} | worst fold {f1s.min():.4f} | "
          f"false alarms mean {fas.mean():.3%}, worst {fas.max():.3%}")

result = pd.DataFrame(summary)
print("\n" + "=" * 90)
print("CONFIGURATION COMPARISON (blocked 5-fold CV)")
print("=" * 90)
print(result)
result.to_csv(OUT_DIR / "stage6_config_comparison.csv")
print("\nSaved: outputs/stage6_config_comparison.csv")
print(f"Total time: {(time.time() - total_start) / 60:.1f} minutes")