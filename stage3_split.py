from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "ddos_cleaned.csv"

df = pd.read_csv(DATA_FILE)
print("Loaded cleaned data:", df.shape)

# --- Separate inputs (X) from the answer we want to predict (y) ---
# Label and Target are NOT inputs: they ARE the answer.
X = df.drop(columns=["Label", "Target"])
y = df["Target"]

print("\nX (input features) shape:", X.shape)
print("y (target) shape        :", y.shape)
print("'Label' in X:", "Label" in X.columns, "| 'Target' in X:", "Target" in X.columns)

# --- Split: 80% training, 20% testing ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% of the data is held back for testing
    random_state=42,     # fixed seed: gives the same split every time
    stratify=y,          # keep the class proportions the same in both sets
)

print("\nTraining set:", X_train.shape, "| Test set:", X_test.shape)

print("\nClass proportions (%):")
print("Full data :", (y.value_counts(normalize=True) * 100).round(2).to_dict())
print("Train set :", (y_train.value_counts(normalize=True) * 100).round(2).to_dict())
print("Test set  :", (y_test.value_counts(normalize=True) * 100).round(2).to_dict())

# --- Sanity checks ---
overlap = set(X_train.index) & set(X_test.index)
print("\nRows in both train and test:", len(overlap))
print("Train + test rows equals total:", len(X_train) + len(X_test) == len(df))
print("Number of feature columns:", X_train.shape[1])