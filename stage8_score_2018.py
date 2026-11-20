import gc
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # save the chart without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import nids_predict as nids
from cic2018_mapping import FEATURE_MAP, LABEL_MAP

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "external_2018"
OUT_DIR = BASE_DIR / "outputs"
FILES = sorted(DATA_DIR.glob("*_TrafficForML_CICFlowMeter.csv"))

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

model, meta = nids.load_model()
features = meta["feature_columns"]
class_names = [meta["classes"][str(i)] for i in range(len(meta["classes"]))]
name_to_id = {n: i for i, n in enumerate(class_names)}
BENIGN_ID = name_to_id["BENIGN"]
RENAME = {v: k for k, v in FEATURE_MAP.items()}
WANTED = set(FEATURE_MAP.values()) | {"Label"}


def load_file(path):
    """Read one file in chunks; return model-named float32 features and mapped labels."""
    xs, ys = [], []
    n_read = n_header = n_unknown = 0
    for chunk in pd.read_csv(path, usecols=lambda c: c.strip() in WANTED, chunksize=100_000):
        chunk.columns = chunk.columns.str.strip()
        n_read += len(chunk)
        label = chunk["Label"].astype(str).str.strip()
        is_header = label == "Label"                   # a stray header row inside the data
        known = label.isin(LABEL_MAP)
        n_header += int(is_header.sum())
        n_unknown += int((~is_header & ~known).sum())
        chunk, label = chunk[known], label[known]
        x = chunk[list(FEATURE_MAP.values())].apply(pd.to_numeric, errors="coerce")
        xs.append(x.rename(columns=RENAME)[features].astype(np.float32))
        ys.append(label.map(LABEL_MAP))
    X = pd.concat(xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    return X, y, n_read, n_header, n_unknown


# ---------------------------------------------------------
# 1. Clean and score each file
# ---------------------------------------------------------
file_rows = []
true_parts, pred_parts, conf_parts, fileid_parts = [], [], [], []
total_start = time.time()

for file_id, path in enumerate(FILES):
    start = time.time()
    short = path.name.split("_")[0]
    X, y, n_read, n_header, n_unknown = load_file(path)

    # exact duplicates, removed the same way as in the training data
    dup = X.assign(_label=y.map(name_to_id).astype(np.int8)).duplicated().to_numpy()
    n_dup = int(dup.sum())
    X = X[~dup].reset_index(drop=True)
    y = y[~dup].reset_index(drop=True)

    # the same prediction pipeline the dashboard uses
    result = nids.predict_flows(X)
    ok = (result["status"] == nids.STATUS_OK).to_numpy()
    n_bad = int((result["status"] == nids.STATUS_BAD_VALUE).sum())
    n_neg = int((result["status"] == nids.STATUS_NEG_TIMING).sum())

    true_id = y[ok].map(name_to_id).to_numpy(dtype=np.int8)
    pred_id = result.loc[ok, "predicted_class"].map(name_to_id).to_numpy(dtype=np.int8)
    conf = result.loc[ok, "confidence"].to_numpy(dtype=np.float32)

    is_benign = true_id == BENIGN_ID
    file_rows.append({
        "file": short,
        "rows_read": n_read,
        "header_or_unknown": n_header + n_unknown,
        "duplicates": n_dup,
        "bad_value": n_bad,
        "negative_timing": n_neg,
        "scored": int(ok.sum()),
        "benign": int(is_benign.sum()),
        "attacks": int((~is_benign).sum()),
        "false_alarm_rate": float((pred_id[is_benign] != BENIGN_ID).mean()) if is_benign.any() else np.nan,
        "attacks_caught": float((pred_id[~is_benign] != BENIGN_ID).mean()) if (~is_benign).any() else np.nan,
    })
    true_parts.append(true_id)
    pred_parts.append(pred_id)
    conf_parts.append(conf)
    fileid_parts.append(np.full(len(true_id), file_id, dtype=np.int8))
    print(f"{short}: done in {time.time() - start:.0f}s | scored {int(ok.sum()):,} flows")

    del X, y, result, dup
    gc.collect()

y_true = np.concatenate(true_parts)
y_pred = np.concatenate(pred_parts)
confidence = np.concatenate(conf_parts)
file_ids = np.concatenate(fileid_parts)

# ---------------------------------------------------------
# 2. Results
# ---------------------------------------------------------
n_cls = len(class_names)
cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
present = [int(c) for c in np.unique(y_true)]
attack_ids = [c for c in present if c != BENIGN_ID]

per_file = pd.DataFrame(file_rows).set_index("file")
print("\n" + "=" * 100)
print("PER FILE (cleaning counts and results)")
print("=" * 100)
print(per_file.round(4).to_string())

total = cm.sum()
accuracy = np.trace(cm) / total
false_alarm = 1 - cm[BENIGN_ID, BENIGN_ID] / cm[BENIGN_ID].sum()
attacks_caught = 1 - cm[attack_ids, BENIGN_ID].sum() / cm[attack_ids].sum()

rows = []
for c in present:
    support = cm[c].sum()
    tp = cm[c, c]
    predicted_total = cm[:, c].sum()
    rows.append({
        "class": class_names[c],
        "flows": int(support),
        "recall": tp / support,
        "precision": tp / predicted_total if predicted_total else np.nan,
        "detected_as_any_attack": (1 - cm[c, BENIGN_ID] / support) if c != BENIGN_ID else np.nan,
    })
per_class = pd.DataFrame(rows).set_index("class")
per_class["f1"] = 2 * per_class["precision"] * per_class["recall"] / (per_class["precision"] + per_class["recall"])
macro_f1 = per_class["f1"].fillna(0).mean()

print("\n" + "=" * 100)
print("OVERALL, all six files together (classes present in the 2018 files only; no PortScan)")
print("=" * 100)
print(f"Flows scored        : {total:,}")
print(f"Accuracy            : {accuracy:.4f}")
print(f"Macro-F1            : {macro_f1:.4f}")
print(f"False-alarm rate    : {false_alarm:.2%}   (benign flows flagged as any attack)")
print(f"Attacks caught      : {attacks_caught:.2%}   (attack flows flagged as any attack, even with the wrong type)")
cv = meta["blocked_cv_estimate_from_experiments"]
print(f"For comparison, blocked CV on CICIDS2017: macro-F1 {cv['macro_f1_mean']:.3f}, "
      f"false alarms {cv['false_alarm_mean']:.2%}, attacks caught 99.35%")

print("\n" + "=" * 100)
print("PER CLASS")
print("=" * 100)
print(per_class.round(3).to_string())

off = cm.copy()
np.fill_diagonal(off, 0)
rows_, cols_ = np.unravel_index(np.argsort(-off.ravel())[:10], off.shape)
print("\nThe 10 biggest mistakes:")
for a, p in zip(rows_, cols_):
    print(f"  actual {class_names[a]:<17} -> predicted {class_names[p]:<17}: {off[a, p]:,} flows")

fa = pd.Series(cm[BENIGN_ID], index=class_names).drop("BENIGN")
print("\nWhere the benign false alarms go:")
print(fa[fa > 0].sort_values(ascending=False).astype(int).to_string())

# ---------------------------------------------------------
# 3. Save everything for the report
# ---------------------------------------------------------
present_names = [class_names[c] for c in present]
pd.DataFrame(cm[present], index=present_names, columns=class_names).to_csv(OUT_DIR / "cross2018_confusion_counts.csv")
per_class.to_csv(OUT_DIR / "cross2018_per_class.csv")
per_file.to_csv(OUT_DIR / "cross2018_per_file.csv")
pd.DataFrame({"file_id": range(len(FILES)), "file": [p.name.split("_")[0] for p in FILES]}).to_csv(
    OUT_DIR / "cross2018_file_index.csv", index=False)
pd.DataFrame({"file_id": file_ids, "true": y_true, "pred": y_pred, "confidence": confidence}).to_parquet(
    OUT_DIR / "cross2018_predictions.parquet")

norm = cm[present] / cm[present].sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(n_cls))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticks(range(len(present)))
ax.set_yticklabels(present_names)
for i in range(len(present)):
    for j in range(n_cls):
        if norm[i, j] >= 0.005:
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
ax.set_xlabel("Predicted class (model trained on CICIDS2017)")
ax.set_ylabel("Actual class (CSE-CIC-IDS2018 label, mapped)")
ax.set_title("Cross-dataset test on CSE-CIC-IDS2018 (rows scaled to 1.0)")
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(OUT_DIR / "cross2018_confusion_normalized.png", dpi=200)
plt.close(fig)

print("\nSaved to outputs/: cross2018_confusion_counts.csv, cross2018_per_class.csv, cross2018_per_file.csv,")
print("  cross2018_file_index.csv, cross2018_predictions.parquet, cross2018_confusion_normalized.png")
print(f"Total time: {(time.time() - total_start) / 60:.1f} minutes")