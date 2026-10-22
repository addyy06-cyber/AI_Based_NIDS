from pathlib import Path
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(DATA_FILE)

# The label column name has a leading space in this file
LABEL = " Label"

print("=" * 60)
print("1. df.info()")
print("=" * 60)
df.info()

print("\n" + "=" * 60)
print("2. df.isnull().sum()  (missing values)")
print("=" * 60)
missing = df.isnull().sum()
print("Total missing values in the whole table:", missing.sum())
print("Columns that have missing values:")
print(missing[missing > 0])

print("\n" + "=" * 60)
print("3. df.duplicated().sum()  (duplicate rows)")
print("=" * 60)
print("Number of duplicate rows:", df.duplicated().sum())

print("\n" + "=" * 60)
print("4. Label counts")
print("=" * 60)
print(df[LABEL].value_counts())