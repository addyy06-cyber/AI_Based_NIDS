import time
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "ddos_cleaned.csv"
MODEL_FILE = BASE_DIR / "outputs" / "rf_model.joblib"

# --- Same loading and splitting as Step 1 (same seed = same split) ---
df = pd.read_csv(DATA_FILE)
X = df.drop(columns=["Label", "Target"])
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training on", X_train.shape[0], "rows and", X_train.shape[1], "features")

# --- Create and train the model ---
model = RandomForestClassifier(
    n_estimators=100,   # number of trees
    n_jobs=-1,          # use all CPU cores to train faster
    random_state=42,    # makes the results repeatable
)

start = time.time()
model.fit(X_train, y_train)    # <- this is the "learning" step
print("Training finished in", round(time.time() - start, 1), "seconds")

# --- What did the model build? ---
print("\nNumber of trees:", len(model.estimators_))
print("Classes the model knows:", list(model.classes_))
print("Features it was trained on:", model.n_features_in_)

# --- Which features mattered most? (based on training only) ---
importances = pd.Series(model.feature_importances_, index=X_train.columns)
print("\nTop 10 most important features:")
print(importances.sort_values(ascending=False).head(10).round(4))

# --- Save the trained model for the next step ---
joblib.dump(model, MODEL_FILE)
print("\nModel saved to:", MODEL_FILE)
print("File size (MB):", round(MODEL_FILE.stat().st_size / 1_000_000, 1))