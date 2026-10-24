from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "all_cleaned.parquet"

# Only the Label column is needed, which keeps memory use small
df = pd.read_parquet(DATA_FILE, columns=["Label"])
print("Index type:", type(df.index).__name__,
      "| first values:", list(df.index[:5]),
      "| increasing:", df.index.is_monotonic_increasing)

labels = df["Label"].astype(str).str.strip()
labels = labels.where(~labels.str.startswith("Web Attack"), "Web Attack")
labels = labels[~labels.isin(["Heartbleed", "Infiltration"])].reset_index(drop=True)
n = len(labels)
print("Rows:", n)

print("\nLabels of the first 5 rows:", labels.head(5).tolist())
print("Labels of the last 5 rows :", labels.tail(5).tolist())

# --- Where does each class sit along the row order? (0% = first row, 100% = last row) ---
pos = pd.Series(np.arange(n))
grp = pos.groupby(labels.values)
summary = pd.DataFrame({"rows": labels.value_counts()})
summary["first_pos_%"] = (grp.min() / n * 100).round(1)
summary["last_pos_%"] = (grp.max() / n * 100).round(1)
print("\nPosition of each class in the row order:")
print(summary.sort_values("first_pos_%"))

# --- Share of each class falling into each tenth of the dataset ---
N_CHUNKS = 10
chunk = np.arange(n) * N_CHUNKS // n          # 0 .. 9
table = pd.crosstab(labels.values, chunk)
share = (table.div(table.sum(axis=1), axis=0) * 100).round(1)
share.columns = [f"{i * 10}-{i * 10 + 10}%" for i in range(N_CHUNKS)]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print("\n% of each class's rows inside each tenth of the dataset:")
print(share)