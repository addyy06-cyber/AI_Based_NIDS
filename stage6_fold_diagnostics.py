import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "all_cleaned.parquet"
MAP_FILE = BASE_DIR / "data" / "processed" / "label_map_model.json"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

N_FOLDS, GAP_FRAC, ATTACK_CAP, BENIGN_CAP = 5, 0.02, 100_000, 300_000
CONFIGS = [
    {"name": "baseline",   "min_samples_leaf": 1},
    {"name": "min_leaf_5", "min_samples_leaf": 5},
]

with open(MAP_FILE) as f:
    raw_map = json.load(f)
label_to_id = dict(raw_map) if all(isinstance(v, int) for v in raw_map.values()) \
    else {name: int(i) for i, name in raw_map.items()}
id_to_label = {i: n for n, i in label_to_id.items()}
class_ids = sorted(id_to_label)
class_names = [id_to_label[i] for i in class_ids]
benign_id = label_to_id["BENIGN"]
bot_pos = class_ids.index(label_to_id["Bot"])

df = pd.read_parquet(DATA_FILE)
labels = df["Label"].astype(str).str.strip()
labels = labels.where(~labels.str.startswith("Web Attack"), "Web Attack")
keep = labels.isin(label_to_id)
feature_cols = [c for c in df.columns if c not in ("Label", "Target")]
X = df.loc[keep, feature_cols].to_numpy(dtype=np.float32)
y = labels[keep].map(label_to_id).to_numpy(dtype=np.int8)
del df, labels, keep
print("Rows used:", len(y), "| Features:", len(feature_cols))

block = np.zeros(len(y), dtype=np.int8)
rank = np.zeros(len(y), dtype=np.int64)
class_size = {}
for c in class_ids:
    idx = np.flatnonzero(y == c)
    r = np.arange(len(idx))
    rank[idx] = r
    block[idx] = r * N_FOLDS // len(idx)
    class_size[c] = len(idx)


def top(values, skip, k=3):
    items = [(class_names[i], int(v)) for i, v in enumerate(values) if i != skip and v > 0]
    items.sort(key=lambda t: -t[1])
    return ", ".join(f"{name}:{v}" for name, v in items[:k]) or "none"


rows = []
for cfg in CONFIGS:
    print(f"\n=== {cfg['name']} ===")
    for fold in range(N_FOLDS):
        test_idx = np.flatnonzero(block == fold)
        parts = []
        for c in class_ids:
            rng = np.random.default_rng([100 + fold, int(c)])
            in_c = (y == c)
            r_test = rank[in_c & (block == fold)]
            lo, hi = r_test.min(), r_test.max()
            gap = max(1, int(GAP_FRAC * class_size[c]))
            idx_c = np.flatnonzero(in_c & ((rank < lo - gap) | (rank > hi + gap)))
            cap = BENIGN_CAP if c == benign_id else ATTACK_CAP
            if len(idx_c) > cap:
                idx_c = rng.choice(idx_c, size=cap, replace=False)
            parts.append(idx_c)
        train_idx = np.concatenate(parts)

        model = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, class_weight="balanced_subsample",
            min_samples_leaf=cfg["min_samples_leaf"], random_state=42)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        y_te = y[test_idx]

        cm = confusion_matrix(y_te, pred, labels=class_ids)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            OUT_DIR / f"stage6_cm_{cfg['name']}_fold{fold + 1}.csv")

        n_benign = int(cm[benign_id].sum())
        false_alarms = n_benign - int(cm[benign_id, benign_id])
        bot_tp = int(cm[bot_pos, bot_pos])
        bot_recall = bot_tp / max(1, int(cm[bot_pos].sum()))
        bot_precision = bot_tp / max(1, int(cm[:, bot_pos].sum()))
        macro = f1_score(y_te, pred, average="macro", labels=class_ids, zero_division=0)
        rows.append({"config": cfg["name"], "fold": fold + 1, "macro_f1": macro,
                     "false_alarm_rate": false_alarms / n_benign, "false_alarms": false_alarms,
                     "bot_recall": bot_recall, "bot_precision": bot_precision})

        print(f"fold {fold + 1}: macro-F1 {macro:.3f} | false alarms {false_alarms}/{n_benign} "
              f"({false_alarms / n_benign:.3%}) | Bot recall {bot_recall:.3f}, precision {bot_precision:.3f}")
        print(f"   BENIGN wrongly labeled as: {top(cm[benign_id], benign_id)}")
        print(f"   Bot flows labeled as     : {top(cm[bot_pos], bot_pos)}")
        print(f"   Predicted Bot came from  : {top(cm[:, bot_pos], bot_pos)}")

pd.DataFrame(rows).to_csv(OUT_DIR / "stage6_fold_diagnostics.csv", index=False)
print("\nSaved: outputs/stage6_fold_diagnostics.csv and 10 confusion-matrix CSVs in outputs/")