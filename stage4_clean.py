from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Load (same as stage4_load.py) ----------
frames = []
for f in sorted(RAW_DIR.glob("*.csv")):
    d = pd.read_csv(f, encoding="latin1", low_memory=False)
    d.columns = d.columns.str.strip()
    d["Label"] = (d["Label"]
                  .str.replace(r"[^\x00-\x7F]+", "-", regex=True)
                  .str.strip())
    num_cols = d.columns.difference(["Label"])
    d[num_cols] = d[num_cols].astype("float32")
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
del frames
df["Label"] = df["Label"].astype("object")
print("Loaded:", df.shape)

start_counts = df["Label"].value_counts()
feature_cols = [c for c in df.columns if c != "Label"]

def show_removed(name, before_df, after_df):
    """Print how many rows of each class a cleaning step removed."""
    removed = before_df["Label"].value_counts() - after_df["Label"].value_counts().reindex(
        before_df["Label"].value_counts().index, fill_value=0)
    removed = removed[removed > 0]
    print(f"\n{name}: removed {len(before_df) - len(after_df):,} rows")
    if len(removed):
        print(removed.to_string())

# ---------- STEP 1: inf / NaN ----------
before = df
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
show_removed("STEP 1 (NaN / inf)", before, df)

# ---------- STEP 2: redundant column ----------
if "Fwd Header Length.1" in df.columns:
    same = df["Fwd Header Length"].equals(df["Fwd Header Length.1"])
    print("\nSTEP 2: 'Fwd Header Length.1' identical to 'Fwd Header Length':", same)
    if same:
        df = df.drop(columns=["Fwd Header Length.1"])

# ---------- STEP 3: exact duplicates ----------
before = df
df = df.drop_duplicates().reset_index(drop=True)
show_removed("STEP 3 (exact duplicates)", before, df)

# ---------- STEP 4: negative timing values ----------
timing_cols = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s",
               "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Min"]
neg_rows = (df[timing_cols] < 0).any(axis=1)
before = df
df = df.loc[~neg_rows].reset_index(drop=True)
show_removed("STEP 4 (negative timing)", before, df)

# ---------- STEP 5: constant columns ----------
feature_cols = [c for c in df.columns if c != "Label"]
nun = df[feature_cols].nunique()
constant_cols = nun[nun == 1].index.tolist()
print("\nSTEP 5: constant columns dropped:", len(constant_cols))
print(constant_cols)
df = df.drop(columns=constant_cols)

# ---------- STEP 6: label encoding ----------
labels_sorted = sorted(df["Label"].unique(), key=lambda x: (x != "BENIGN", x))
label_map = {name: i for i, name in enumerate(labels_sorted)}
df["Target"] = df["Label"].map(label_map).astype("int8")
with open(OUT_DIR / "label_map.json", "w") as fh:
    json.dump(label_map, fh, indent=2)
print("\nSTEP 6: label map")
print(json.dumps(label_map, indent=2))

# ---------- Summary: before vs after ----------
end_counts = df["Label"].value_counts()
summary = pd.DataFrame({"before": start_counts, "after": end_counts}).fillna(0).astype(int)
summary["kept_%"] = (summary["after"] / summary["before"] * 100).round(1)
print("\nPER-CLASS SUMMARY (before -> after cleaning)")
print(summary.sort_values("after", ascending=False).to_string())
print("\nFinal shape:", df.shape)

# ---------- Save ----------
try:
    out = OUT_DIR / "all_cleaned.parquet"
    df.to_parquet(out, index=False)
except Exception as e:
    print("\nParquet not available (", type(e).__name__, "), saving CSV instead")
    out = OUT_DIR / "all_cleaned.csv"
    df.to_csv(out, index=False)
print("Saved to:", out, "| size (MB):", round(out.stat().st_size / 1e6, 1))

# ---------- Verify ----------
check = pd.read_parquet(out) if out.suffix == ".parquet" else pd.read_csv(out)
num = check.select_dtypes("number")
checks = [
    ("Shape matches",            check.shape == df.shape),
    ("No missing values",        check.isnull().sum().sum() == 0),
    ("No infinite values",       np.isinf(num).sum().sum() == 0),
    ("No duplicate rows",        check.duplicated().sum() == 0),
    ("Every label has a Target", check["Target"].notnull().all()),
    ("Label/Target agree",       (check["Label"].map(label_map) == check["Target"]).all()),
]
print("\nVERIFICATION")
for desc, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + desc)