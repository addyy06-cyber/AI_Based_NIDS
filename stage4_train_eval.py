from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, accuracy_score)

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

train = pd.read_parquet(PROC_DIR / "train_sampled.parquet")
test = pd.read_parquet(PROC_DIR / "test_full.parquet")
with open(PROC_DIR / "label_map_model.json") as fh:
    label_map = json.load(fh)
names = [n for n, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

X_train = train.drop(columns=["Label", "Target"])
y_train = train["Target"]
X_test = test.drop(columns=["Label", "Target"])
y_test = test["Target"]
print("Train:", X_train.shape, "| Test:", X_test.shape)

# ---------- Train ----------
model = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced_subsample",   # gives rare classes more weight
)
t0 = time.time()
model.fit(X_train, y_train)
print("Training time: %.1f s" % (time.time() - t0))

joblib.dump(model, OUT_DIR / "rf_multiclass.joblib", compress=3)
print("Model size (MB): %.1f" % ((OUT_DIR / "rf_multiclass.joblib").stat().st_size / 1e6))

# ---------- Predict ----------
y_pred = model.predict(X_test)

print("\nAccuracy : %.4f" % accuracy_score(y_test, y_pred))
print("Macro-F1 : %.4f" % f1_score(y_test, y_pred, average="macro"))

print("\nCLASSIFICATION REPORT (test set)")
print(classification_report(y_test, y_pred, target_names=names, digits=4))

# ---------- Binary view: BENIGN vs any attack ----------
actual_attack = (y_test != 0)
pred_attack = (y_pred != 0)
tp = int((actual_attack & pred_attack).sum())
fn = int((actual_attack & ~pred_attack).sum())
fp = int((~actual_attack & pred_attack).sum())
tn = int((~actual_attack & ~pred_attack).sum())
print("BINARY VIEW (BENIGN vs any attack)")
print("Attacks caught (any type) :", tp, "of", tp + fn, "(%.2f%%)" % (tp / (tp + fn) * 100))
print("Attacks missed as BENIGN  :", fn)
print("False alarms on BENIGN    :", fp, "of", fp + tn, "(%.3f%%)" % (fp / (fp + tn) * 100))

# ---------- Confusion matrix (normalized by row) ----------
cm = confusion_matrix(y_test, y_pred)
print("\nRaw confusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(cm, index=names, columns=range(len(names))).to_string())
print("Column numbers:", {i: n for i, n in enumerate(names)})

fig, ax = plt.subplots(figsize=(11, 9))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=names, normalize="true",
    values_format=".2f", xticks_rotation=45, cmap="Blues", ax=ax, colorbar=False
)
ax.set_title("Normalized Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_multiclass.png", dpi=150)
print("\nChart saved to:", OUT_DIR / "confusion_matrix_multiclass.png")

# ---------- Top features ----------
imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTOP 10 FEATURES")
print(imp.head(10).round(4).to_string())