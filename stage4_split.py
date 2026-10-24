from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"

# ---------- Settings you can change ----------
DROP_CLASSES = ["Heartbleed", "Infiltration"]   # too few rows to train/test reliably
MERGE_WEB_ATTACKS = True                        # merge the 3 web attack labels into one
CAP_BENIGN = 300_000                            # max BENIGN rows in the TRAINING set
CAP_OTHER = 100_000                             # max rows per attack class in the TRAINING set
SEED = 42
# ---------------------------------------------

df = pd.read_parquet(PROC_DIR / "all_cleaned.parquet")
df = df.drop(columns="Target")                  # rebuilt below for the new label set
print("Loaded:", df.shape)

# 1. Drop tiny classes
if DROP_CLASSES:
    df = df[~df["Label"].isin(DROP_CLASSES)]
    print("Dropped classes:", DROP_CLASSES)

# 2. Merge web attacks
if MERGE_WEB_ATTACKS:
    df["Label"] = df["Label"].where(~df["Label"].str.startswith("Web Attack"), "Web Attack")
df = df.reset_index(drop=True)

# 3. New label map (saved separately so the old one is not overwritten)
labels = sorted(df["Label"].unique(), key=lambda x: (x != "BENIGN", x))
label_map = {name: i for i, name in enumerate(labels)}
df["Target"] = df["Label"].map(label_map).astype("int8")
with open(PROC_DIR / "label_map_model.json", "w") as fh:
    json.dump(label_map, fh, indent=2)
print("\nLabel map:", json.dumps(label_map, indent=2))

# 4. Stratified split on the FULL data (split row positions, which saves memory)
idx = np.arange(len(df))
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, random_state=SEED, stratify=df["Target"]
)
train = df.iloc[train_idx]
test = df.iloc[test_idx].reset_index(drop=True)

# 5. Cap the large classes in the TRAINING set only
parts = []
for label, grp in train.groupby("Label"):
    cap = CAP_BENIGN if label == "BENIGN" else CAP_OTHER
    parts.append(grp.sample(n=min(len(grp), cap), random_state=SEED))
train_s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

# 6. Summary
summary = pd.DataFrame({
    "all_rows":   df["Label"].value_counts(),
    "train_full": train["Label"].value_counts(),
    "train_used": train_s["Label"].value_counts(),
    "test":       test["Label"].value_counts(),
}).fillna(0).astype(int)
print("\nPER-CLASS COUNTS")
print(summary.sort_values("all_rows", ascending=False).to_string())

print("\nTotals -> all: {:,} | train used: {:,} | test: {:,}".format(
    len(df), len(train_s), len(test)))
share = lambda d: (d["Label"] == "BENIGN").mean() * 100
print("BENIGN share -> all: %.2f%% | train used: %.2f%% | test: %.2f%%" % (
    share(df), share(train_s), share(test)))

# 7. Save
train_s.to_parquet(PROC_DIR / "train_sampled.parquet", index=False)
test.to_parquet(PROC_DIR / "test_full.parquet", index=False)
print("\nSaved train_sampled.parquet and test_full.parquet in", PROC_DIR)