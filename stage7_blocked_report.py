import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # save the figure without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
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
assert class_ids == list(range(len(class_ids)))
class_names = [id_to_label[i] for i in class_ids]
n_classes = len(class_ids)
benign_id = label_to_id["BENIGN"]

# --- Data (row order = capture order) ---
df = pd.read_parquet(DATA_FILE)
labels = df["Label"].astype(str).str.strip()
labels = labels.where(~labels.str.startswith("Web Attack"), "Web Attack")
keep = labels.isin(label_to_id)
feature_cols = [c for c in df.columns if c not in ("Label", "Target")]
X = df.loc[keep, feature_cols].to_numpy(dtype=np.float32)
y = labels[keep].map(label_to_id).to_numpy(dtype=np.int64)
del df, labels, keep
print("Rows used:", len(y), "| Features:", X.shape[1])

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

# --- Blocked CV with the FINAL configuration ---
overall_rows, rec_rows, prec_rows, f1_rows = [], [], [], []
cm_total = np.zeros((n_classes, n_classes), dtype=np.int64)
total_start = time.time()

for fold in range(N_FOLDS):
    start = time.time()
    rng = np.random.default_rng(100 + fold)
    test_idx = np.flatnonzero(block == fold)

    parts = []
    for c in class_ids:
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

    # sqrt class weights, computed on THIS fold's training set
    counts = np.bincount(y[train_idx], minlength=n_classes)
    weights = {int(c): float(np.sqrt(len(train_idx) / (n_classes * counts[c])))
               for c in class_ids}

    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                   class_weight=weights, n_jobs=-1, random_state=42)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    y_te = y[test_idx]

    cm = confusion_matrix(y_te, pred, labels=class_ids)
    cm_total += cm

    true_attack = y_te != benign_id
    pred_attack = pred != benign_id
    n_false_alarms = int((~true_attack & pred_attack).sum())
    overall_rows.append({
        "fold": fold + 1,
        "test_rows": len(y_te),
        "benign_test_rows": int((~true_attack).sum()),
        "accuracy": accuracy_score(y_te, pred),
        "macro_f1": f1_score(y_te, pred, average="macro", labels=class_ids, zero_division=0),
        "attacks_caught": (true_attack & pred_attack).sum() / true_attack.sum(),
        "false_alarms": n_false_alarms,
        "false_alarm_rate": n_false_alarms / (~true_attack).sum(),
    })
    rec_rows.append(recall_score(y_te, pred, average=None, labels=class_ids, zero_division=0))
    prec_rows.append(precision_score(y_te, pred, average=None, labels=class_ids, zero_division=0))
    f1_rows.append(f1_score(y_te, pred, average=None, labels=class_ids, zero_division=0))
    print(f"Fold {fold + 1}/{N_FOLDS}: macro-F1={overall_rows[-1]['macro_f1']:.4f}  "
          f"false alarms={n_false_alarms:,}  ({time.time() - start:.0f}s)")

# --- Tables ---
overall = pd.DataFrame(overall_rows)
rec, prec, f1a = np.array(rec_rows), np.array(prec_rows), np.array(f1_rows)
support = cm_total.sum(axis=1)
pooled_recall = np.diag(cm_total) / support
pooled_precision = np.diag(cm_total) / np.maximum(cm_total.sum(axis=0), 1)

per_class = pd.DataFrame({
    "test_rows": support,
    "recall_mean": rec.mean(axis=0), "recall_worst_fold": rec.min(axis=0),
    "precision_mean": prec.mean(axis=0), "precision_worst_fold": prec.min(axis=0),
    "f1_mean": f1a.mean(axis=0), "f1_worst_fold": f1a.min(axis=0),
    "pooled_recall": pooled_recall, "pooled_precision": pooled_precision,
}, index=class_names)

cols = [f"fold{i}" for i in range(1, N_FOLDS + 1)]
overall.to_csv(OUT_DIR / "report_overall_by_fold.csv", index=False)
per_class.to_csv(OUT_DIR / "report_per_class.csv")
pd.DataFrame(rec.T, index=class_names, columns=cols).to_csv(OUT_DIR / "report_recall_by_fold.csv")
pd.DataFrame(prec.T, index=class_names, columns=cols).to_csv(OUT_DIR / "report_precision_by_fold.csv")
pd.DataFrame(cm_total, index=class_names, columns=class_names).to_csv(OUT_DIR / "report_confusion_counts.csv")

# --- Heatmap (each row scaled to 100%, so the diagonal is the recall) ---
norm = cm_total / cm_total.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(n_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticks(range(n_classes))
ax.set_yticklabels(class_names)
for i in range(n_classes):
    for j in range(n_classes):
        if norm[i, j] >= 0.005:
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
ax.set_title("Blocked 5-fold CV, final configuration\n"
             "(rows scaled to 1.0; each flow is tested exactly once)")
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(OUT_DIR / "report_confusion_normalized.png", dpi=200)
plt.close(fig)

# --- Print the summary ---
print("\n" + "=" * 80)
print("OVERALL BY FOLD")
print("=" * 80)
print(overall.round(4).to_string(index=False))
print("\nMean +/- std across folds:")
for m in ("macro_f1", "accuracy", "attacks_caught", "false_alarm_rate"):
    print(f"  {m:<17}: {overall[m].mean():.4f} +/- {overall[m].std(ddof=1):.4f}"
          f"   (worst fold: {overall[m].min() if m != 'false_alarm_rate' else overall[m].max():.4f})")
pooled_f1 = 2 * pooled_precision * pooled_recall / np.maximum(pooled_precision + pooled_recall, 1e-12)
print(f"  pooled macro-F1 (all folds as one test set): {pooled_f1.mean():.4f}")

print("\n" + "=" * 80)
print("PER CLASS")
print("=" * 80)
print(per_class.round(3))

off = cm_total.copy()
np.fill_diagonal(off, 0)
rows_, cols_ = np.unravel_index(np.argsort(-off.ravel())[:8], off.shape)
print("\nThe 8 biggest mistakes (all folds combined):")
for a, p in zip(rows_, cols_):
    print(f"  actual {class_names[a]:<17} -> predicted {class_names[p]:<17}: {off[a, p]:,} flows")

print("\nWhere the BENIGN false alarms go (all folds combined):")
fa = pd.Series(cm_total[benign_id], index=class_names).drop("BENIGN")
print(fa[fa > 0].sort_values(ascending=False).astype(int).to_string())

print("\nSaved to outputs/: report_overall_by_fold.csv, report_per_class.csv,")
print("  report_recall_by_fold.csv, report_precision_by_fold.csv,")
print("  report_confusion_counts.csv, report_confusion_normalized.png")
print(f"Total time: {(time.time() - total_start) / 60:.1f} minutes")