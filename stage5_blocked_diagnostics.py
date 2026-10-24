import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_score, recall_score)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "all_cleaned.parquet"
MAP_FILE = BASE_DIR / "data" / "processed" / "label_map_model.json"
OUT_DIR = BASE_DIR / "outputs"

N_FOLDS = 5
GAP_FRAC = 0.02
BENIGN_CAP = 300_000
ATTACK_CAP = 100_000

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

# --- Data (row order = capture order) ---
df = pd.read_parquet(DATA_FILE)
labels = df["Label"].astype(str).str.strip()
labels = labels.where(~labels.str.startswith("Web Attack"), "Web Attack")
keep = labels.isin(label_to_id)
feature_cols = [c for c in df.columns if c not in ("Label", "Target")]
X = df.loc[keep, feature_cols].to_numpy(dtype=np.float32)
y = labels[keep].map(label_to_id).to_numpy(dtype=np.int8)
del df, labels, keep
print("Rows used:", len(y))

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

# --- Same blocked CV as before, keeping the predictions ---
fold_macro, fold_recall, fold_precision, fold_f1, fold_cm = [], [], [], [], []
saved = {}

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
        cap = BENIGN_CAP if c == benign_id else ATTACK_CAP
        if len(idx_c) > cap:
            idx_c = rng.choice(idx_c, size=cap, replace=False)
        train_parts.append(idx_c)
    train_idx = np.concatenate(train_parts)

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                   class_weight="balanced_subsample", random_state=42)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    y_te = y[test_idx]

    fold_macro.append(f1_score(y_te, pred, average="macro", labels=class_ids, zero_division=0))
    fold_recall.append(recall_score(y_te, pred, average=None, labels=class_ids, zero_division=0))
    fold_precision.append(precision_score(y_te, pred, average=None, labels=class_ids, zero_division=0))
    fold_f1.append(f1_score(y_te, pred, average=None, labels=class_ids, zero_division=0))
    fold_cm.append(confusion_matrix(y_te, pred, labels=class_ids))
    saved[f"test_idx_{fold + 1}"] = test_idx
    saved[f"y_true_{fold + 1}"] = y_te
    saved[f"y_pred_{fold + 1}"] = pred
    print(f"Fold {fold + 1}/{N_FOLDS}: macro-F1={fold_macro[-1]:.4f} ({time.time() - start:.0f}s)")

np.savez_compressed(OUT_DIR / "blocked_cv_predictions.npz", **saved)

# --- Per-fold tables ---
cols = [f"fold{i}" for i in range(1, N_FOLDS + 1)]
for title, data in [("RECALL", fold_recall), ("PRECISION", fold_precision), ("F1", fold_f1)]:
    print("\n" + "=" * 70)
    print(f"{title} BY CLASS AND FOLD")
    print("=" * 70)
    print(pd.DataFrame(np.array(data).T, index=class_names, columns=cols).round(3))

# --- The two worst folds: confusion matrix and biggest mistakes ---
worst = sorted(np.argsort(fold_macro)[:2])
row_labels = [f"{i} {n}" for i, n in zip(class_ids, class_names)]
for k in worst:
    print("\n" + "=" * 70)
    print(f"FOLD {k + 1}: CONFUSION MATRIX (rows = actual, columns = predicted class id)")
    print("=" * 70)
    print(pd.DataFrame(fold_cm[k], index=row_labels, columns=class_ids))

    off = fold_cm[k].copy()
    np.fill_diagonal(off, 0)
    rows, colsx = np.unravel_index(np.argsort(-off.ravel())[:8], off.shape)
    print(f"\nFold {k + 1}: the 8 biggest mistakes")
    for a, p in zip(rows, colsx):
        print(f"  actual {class_names[a]:<17} -> predicted {class_names[p]:<17}: {off[a, p]:,} flows")

print("\nSaved: outputs/blocked_cv_predictions.npz")