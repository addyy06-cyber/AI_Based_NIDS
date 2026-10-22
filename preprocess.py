from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(DATA_FILE)
print("Loaded:", df.shape)

# ----------------------------------------------------------
# STEP 1: Clean column names (remove leading/trailing spaces)
# ----------------------------------------------------------
cols_with_spaces_before = sum(col != col.strip() for col in df.columns)

df.columns = df.columns.str.strip()

cols_with_spaces_after = sum(col != col.strip() for col in df.columns)

print("\nSTEP 1: Column names")
print("Columns with extra spaces before:", cols_with_spaces_before)
print("Columns with extra spaces after :", cols_with_spaces_after)
print("All column names unique         :", df.columns.is_unique)
print("'Label' column now accessible   :", "Label" in df.columns)
print("First 5 names:", list(df.columns[:5]))
print("Last 3 names :", list(df.columns[-3:]))

# ----------------------------------------------------------
# STEP 2: Handle missing and infinite values
# ----------------------------------------------------------
print("\nSTEP 2: Missing and infinite values")

feature_cols = [c for c in df.columns if c != "Label"]

# Find the rows with a NaN or an infinite value in any feature column
nan_rows = df[feature_cols].isnull().any(axis=1)
inf_rows = np.isinf(df[feature_cols]).any(axis=1)
bad_rows = nan_rows | inf_rows

print("Rows with NaN          :", nan_rows.sum())
print("Rows with inf          :", inf_rows.sum())
print("Rows with either (bad) :", bad_rows.sum())

print("\nLabels of the bad rows:")
print(df.loc[bad_rows, "Label"].value_counts())

print("\nA few bad rows:")
print(df.loc[bad_rows, ["Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Label"]].head())

# Fix: turn inf into NaN, then drop every row that has a NaN
rows_before = len(df)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.reset_index(drop=True)
rows_after = len(df)

print("\nRows before:", rows_before)
print("Rows after :", rows_after)
print("Rows removed:", rows_before - rows_after)
print("NaN remaining:", int(df.isnull().sum().sum()))
print("inf remaining:", int(np.isinf(df[feature_cols]).sum().sum()))
print("\nLabel counts after cleaning:")
print(df["Label"].value_counts())

# ----------------------------------------------------------
# STEP 3: Redundant column and duplicate rows
# ----------------------------------------------------------
print("\nSTEP 3: Redundant column and duplicates")

# 3a. Drop the redundant column (only if it is really identical)
identical = df["Fwd Header Length"].equals(df["Fwd Header Length.1"])
print("Fwd Header Length columns identical:", identical)
if identical:
    df = df.drop(columns=["Fwd Header Length.1"])
print("Columns now:", df.shape[1])

# 3b. Look at the duplicates first
dup_mask = df.duplicated()
print("\nDuplicate rows:", dup_mask.sum())
print("Percentage of data:", round(dup_mask.mean() * 100, 2), "%")
print("Labels of duplicate rows:")
print(df.loc[dup_mask, "Label"].value_counts())

# 3c. Check for conflicting rows (same features, different label)
feature_cols = [c for c in df.columns if c != "Label"]
unique_full = len(df.drop_duplicates())
unique_features_only = len(df.drop_duplicates(subset=feature_cols))
print("\nConflicting rows (same features, different label):",
      unique_full - unique_features_only)

# 3d. Drop the exact duplicates
rows_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print("\nRows before:", rows_before)
print("Rows after :", len(df))
print("Duplicates remaining:", df.duplicated().sum())
print("\nLabel counts after removing duplicates:")
print(df["Label"].value_counts())
print("\nFinal shape so far:", df.shape)

# ----------------------------------------------------------
# STEP 4: Negative values, constant columns, label encoding
# ----------------------------------------------------------
print("\nSTEP 4: Final checks and label encoding")

feature_cols = [c for c in df.columns if c != "Label"]

# 4a. Look for negative values (report only, nothing is changed)
negatives = (df[feature_cols] < 0).sum()
negatives = negatives[negatives > 0]
print("\nColumns containing negative values (count of rows):")
print(negatives if len(negatives) > 0 else "None")

# 4b. Find and drop constant columns (same value in every row)
unique_counts = df[feature_cols].nunique()
constant_cols = unique_counts[unique_counts == 1].index.tolist()
print("\nConstant columns found:", len(constant_cols))
print(constant_cols)
df = df.drop(columns=constant_cols)

# 4c. Label encoding: text -> number
label_map = {"BENIGN": 0, "DDoS": 1}
df["Target"] = df["Label"].map(label_map)

print("\nTarget column created. Any unmapped labels (NaN):", df["Target"].isnull().sum())
print("Label vs Target counts:")
print(df.groupby(["Label", "Target"]).size())

# 4d. Summary of the cleaned data
print("\nFinal shape:", df.shape)
print("Feature columns:", df.shape[1] - 2, "(excluding Label and Target)")
print("Class balance (%):")
print((df["Target"].value_counts(normalize=True) * 100).round(2))

# ----------------------------------------------------------
# STEP 5: Remove invalid timing rows, save, and verify
# ----------------------------------------------------------
print("\nSTEP 5: Negative timing values, save, verify")

# 5a. Rows with a negative timing/rate value (Init_Win_bytes_* are NOT included: -1 is normal there)
timing_cols = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s",
               "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Min"]
neg_rows = (df[timing_cols] < 0).any(axis=1)
print("Rows with a negative timing value:", neg_rows.sum())
print("Their labels:")
print(df.loc[neg_rows, "Label"].value_counts())

rows_before = len(df)
df = df.loc[~neg_rows].reset_index(drop=True)
print("Rows before:", rows_before, "| after:", len(df))

# 5b. Save the cleaned dataset to a NEW file
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "ddos_cleaned.csv"
df.to_csv(OUTPUT_FILE, index=False)
print("\nSaved to:", OUTPUT_FILE)
print("File size (MB):", round(OUTPUT_FILE.stat().st_size / 1_000_000, 1))

# 5c. Reload the saved file and verify it
check = pd.read_csv(OUTPUT_FILE)
numeric = check.select_dtypes(include="number")

verification = [
    ("Shape matches the cleaned data",   check.shape == df.shape),
    ("Column names match",               list(check.columns) == list(df.columns)),
    ("No missing values",                check.isnull().sum().sum() == 0),
    ("No infinite values",               np.isinf(numeric).sum().sum() == 0),
    ("No duplicate rows",                check.duplicated().sum() == 0),
    ("No spaces in column names",        all(c == c.strip() for c in check.columns)),
    ("Target contains only 0 and 1",     set(check["Target"].unique()) == {0, 1}),
    ("Label and Target agree",           (check["Label"].map({"BENIGN": 0, "DDoS": 1}) == check["Target"]).all()),
]

print("\nSTAGE 2 VERIFICATION")
print("-" * 50)
for description, passed in verification:
    print(("PASS  " if passed else "FAIL  ") + description)
print("-" * 50)
print("Final shape:", check.shape)
print(check["Label"].value_counts())
print("\nAll checks passed!" if all(p for _, p in verification)
      else "\nSome checks FAILED - tell me which ones.")