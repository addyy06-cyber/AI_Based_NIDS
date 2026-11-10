import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "all_cleaned.parquet"
MAP_FILE = BASE_DIR / "data" / "processed" / "label_map_model.json"
OUT_DIR = BASE_DIR / "outputs"

BENIGN_CAP = 300_000
ATTACK_CAP = 100_000
SEED = 42

# ---------------------------------------------------------
# 1. Class map
# ---------------------------------------------------------
with open(MAP_FILE) as f:
    raw_map = json.load(f)
if all(isinstance(v, int) for v in raw_map.values()):
    label_to_id = dict(raw_map)
else:
    label_to_id = {name: int(i) for i, name in raw_map.items()}
id_to_label = {i: n for n, i in label_to_id.items()}
class_ids = sorted(id_to_label)
assert class_ids == list(range(len(class_ids))), "class ids must be 0..N-1"
benign_id = label_to_id["BENIGN"]

# ---------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------
df = pd.read_parquet(DATA_FILE)
labels = df["Label"].astype(str).str.strip()
labels = labels.where(~labels.str.startswith("Web Attack"), "Web Attack")
keep = labels.isin(label_to_id)

feature_cols = [c for c in df.columns if c not in ("Label", "Target")]
X_all = df.loc[keep, feature_cols].astype(np.float32).reset_index(drop=True)
y_all = labels[keep].map(label_to_id).to_numpy(dtype=np.int64)
del df, labels, keep
print("Rows used:", len(y_all), "| Features:", len(feature_cols))

# ---------------------------------------------------------
# 3. Build the training set (same caps as the experiments)
# ---------------------------------------------------------
rng = np.random.default_rng(SEED)
parts = []
for c in class_ids:
    idx_c = np.flatnonzero(y_all == c)
    cap = BENIGN_CAP if c == benign_id else ATTACK_CAP
    if len(idx_c) > cap:
        idx_c = rng.choice(idx_c, size=cap, replace=False)
    parts.append(idx_c)
train_idx = np.sort(np.concatenate(parts))
X_train = X_all.iloc[train_idx]
y_train = y_all[train_idx]

counts = np.bincount(y_train, minlength=len(class_ids))
n_train = len(y_train)
class_weight = {int(c): float(np.sqrt(n_train / (len(class_ids) * counts[c])))
                for c in class_ids}

print("\nTraining rows:", n_train)
table = pd.DataFrame({
    "training_rows": counts,
    "weight": [round(class_weight[c], 3) for c in class_ids],
}, index=[id_to_label[c] for c in class_ids])
print(table)

# ---------------------------------------------------------
# 4. Train
# ---------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100, min_samples_leaf=5,
    class_weight=class_weight, n_jobs=-1, random_state=SEED,
)
start = time.time()
model.fit(X_train, y_train)
print(f"\nTraining finished in {time.time() - start:.0f} seconds")

# ---------------------------------------------------------
# 5. Save the model and everything needed to use it
# ---------------------------------------------------------
MODEL_FILE = OUT_DIR / "final_model.joblib"
joblib.dump(model, MODEL_FILE)

with open(OUT_DIR / "final_label_map.json", "w") as f:
    json.dump(label_to_id, f, indent=2)

meta = {
    "trained_on": time.strftime("%Y-%m-%d %H:%M"),
    "data_file": "data/processed/all_cleaned.parquet (CICIDS2017, 11 classes)",
    "model": "RandomForestClassifier",
    "params": {"n_estimators": 100, "min_samples_leaf": 5, "random_state": SEED,
               "class_weight": "sqrt(n_train / (n_classes * count_c)), computed on the training set"},
    "class_weight_values": {id_to_label[c]: round(class_weight[c], 4) for c in class_ids},
    "caps": {"BENIGN": BENIGN_CAP, "per_attack_class": ATTACK_CAP},
    "training_rows_per_class": {id_to_label[c]: int(counts[c]) for c in class_ids},
    "classes": {str(c): id_to_label[c] for c in class_ids},
    "benign_id": benign_id,
    "feature_columns": feature_cols,
    "input_dtype": "float32",
    "preprocessing_required": [
        "strip spaces from column names",
        "replace inf/-inf with NaN and handle or drop those rows",
        "keep exactly the feature_columns above, in this order",
        "convert to float32",
    ],
    "library_versions": {"python": platform.python_version(),
                         "scikit-learn": sklearn.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__},
    "blocked_cv_estimate_from_experiments": {
        "macro_f1_mean": 0.948, "macro_f1_worst_fold": 0.901,
        "false_alarm_mean": 0.0024, "false_alarm_worst_fold": 0.0037,
        "caveat": "Configuration was chosen on these same folds, so the numbers are optimistic. "
                  "Not a test on an untouched dataset or a different network.",
    },
}
with open(OUT_DIR / "final_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved:")
for name in ("final_model.joblib", "final_label_map.json", "final_model_meta.json"):
    p = OUT_DIR / name
    print(f"  outputs/{name}  ({p.stat().st_size / 1e6:.1f} MB)")

# ---------------------------------------------------------
# 6. Checks on the saved files
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("CHECKS ON THE SAVED MODEL")
print("=" * 60)
loaded = joblib.load(MODEL_FILE)

sample = np.random.default_rng(0).choice(len(y_all), size=100_000, replace=False)
Xs = X_all.iloc[sample]
same = np.array_equal(model.predict(Xs), loaded.predict(Xs))
print("Reloaded model gives identical predictions (100,000 rows):", same)
print("Feature names stored in the model match the saved list   :",
      list(loaded.feature_names_in_) == feature_cols)
print("Number of features:", loaded.n_features_in_, "| classes:", list(loaded.classes_))

try:
    loaded.predict(Xs.iloc[:5, ::-1])      # same columns, reversed order
    print("Reversed column order: NOT caught (the pipeline must enforce the order itself)")
except ValueError:
    print("Reversed column order: rejected with an error (good, order mistakes are caught)")

print("\nTop 10 most important features (final model):")
importances = pd.Series(model.feature_importances_, index=feature_cols)
print(importances.sort_values(ascending=False).head(10).round(4))