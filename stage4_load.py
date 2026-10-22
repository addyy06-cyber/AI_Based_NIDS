from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data"          # folder holding the 8 raw CSVs (NOT searched recursively)

files = sorted(RAW_DIR.glob("*.csv"))
print("Found", len(files), "CSV files\n")

frames = []
for f in files:
    d = pd.read_csv(f, encoding="latin1", low_memory=False)
    d.columns = d.columns.str.strip()
    d["Label"] = (d["Label"]
                  .str.replace(r"[^\x00-\x7F]+", "-", regex=True)
                  .str.strip())
    num_cols = d.columns.difference(["Label"])
    d[num_cols] = d[num_cols].astype("float32")
    d["SourceFile"] = f.name
    print(f"{f.name:55s} rows={len(d):>9,}  cols={d.shape[1]}")
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
del frames
df["Label"] = df["Label"].astype("category")
df["SourceFile"] = df["SourceFile"].astype("category")

print("\nTotal rows   :", f"{len(df):,}")
print("Total columns:", df.shape[1])
print("Memory used  : %.2f GB" % (df.memory_usage(deep=True).sum() / 1e9))

print("\nLabel counts:")
counts = df["Label"].value_counts()
print(pd.DataFrame({"rows": counts, "percent": (counts / len(df) * 100).round(3)}).to_string())

num = df.select_dtypes("number")
print("\nMissing (NaN) values :", int(num.isna().sum().sum()))
print("Infinite values      :", int(np.isinf(num).sum().sum()))
print("Exact duplicate rows :", int(df.drop(columns="SourceFile").duplicated().sum()))