from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "ddos_cleaned.csv"
MODEL_FILE = BASE_DIR / "outputs" / "rf_model.joblib"
CHART_FILE = BASE_DIR / "outputs" / "confusion_matrix.png"

# --- Recreate the exact same split (same seed) and load the saved model ---
df = pd.read_csv(DATA_FILE)
X = df.drop(columns=["Label", "Target"])
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = joblib.load(MODEL_FILE)

# --- Predict on data the model has never seen ---
y_pred = model.predict(X_test)

# --- 1. Accuracy (train vs test, to check for overfitting) ---
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
print("Accuracy on training data:", round(train_acc * 100, 3), "%")
print("Accuracy on test data    :", round(test_acc * 100, 3), "%")

# --- 2. Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(cm)
print("\nBENIGN correctly identified (true negatives) :", tn)
print("BENIGN wrongly flagged as DDoS (false alarms):", fp)
print("DDoS wrongly called BENIGN (MISSED attacks)  :", fn)
print("DDoS correctly detected (true positives)     :", tp)

# --- 3. Precision, recall, F1 ---
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["BENIGN", "DDoS"], digits=4))

# --- 4. Save the confusion matrix as a chart ---
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["BENIGN", "DDoS"], values_format="d", ax=ax
)
ax.set_title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig(CHART_FILE, dpi=150)
print("Chart saved to:", CHART_FILE)
plt.show()