from pathlib import Path
import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
LABEL = " Label"

df = pd.read_csv(DATA_FILE)
numeric = df.select_dtypes(include="number")

checks = [
    ("File loaded and is not empty",            len(df) > 0),
    ("Has 79 columns (78 features + label)",    df.shape[1] == 79),
    ("Label column exists",                     LABEL in df.columns),
    ("Label has exactly 2 classes",             df[LABEL].nunique() == 2),
    ("Classes are BENIGN and DDoS",             set(df[LABEL].unique()) == {"BENIGN", "DDoS"}),
    ("No missing labels",                       df[LABEL].isnull().sum() == 0),
    ("All 78 features are numeric",             numeric.shape[1] == 78),
]

print("STAGE 1 VERIFICATION")
print("-" * 50)
for description, passed in checks:
    print(("PASS  " if passed else "FAIL  ") + description)

print("-" * 50)
print("Rows:", len(df))
print("Missing values:", int(df.isnull().sum().sum()))
print("Infinite values:", int(np.isinf(numeric).sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("\nAll checks passed!" if all(p for _, p in checks)
      else "\nSome checks FAILED - tell me which ones.")