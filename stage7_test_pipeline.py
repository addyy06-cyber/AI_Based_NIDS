import time
from pathlib import Path

import numpy as np
import pandas as pd

import nids_predict as nids

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_FILE = BASE_DIR / "data" / "processed" / "all_cleaned.parquet"
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "outputs"

pd.set_option("display.width", 200)

model, meta = nids.load_model()
features = meta["feature_columns"]
class_names = {int(k): v for k, v in meta["classes"].items()}

# ---------------------------------------------------------
print("=" * 70)
print("TEST 1: does the pipeline reproduce direct model predictions?")
print("=" * 70)
sample = pd.read_parquet(CLEAN_FILE, columns=features).sample(n=50_000, random_state=0)
X_sample = sample.astype(np.float32).reset_index(drop=True)
direct = pd.Series(model.predict(X_sample)).map(class_names)

tmp_csv = OUT_DIR / "test_flows_cleaned.csv"
X_sample.to_csv(tmp_csv, index=False)
result1 = nids.predict_flows(nids.read_flows(tmp_csv))

print(result1["status"].value_counts().to_string())
agree = (result1["predicted_class"] == direct).mean()
print(f"Predictions identical to the model's own: {agree:.4%}")

# ---------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 2: shuffled columns, extra spaces in names, extra junk column")
print("=" * 70)
messy = X_sample.sample(frac=1, axis=1, random_state=1)
messy["Junk Column"] = 1
messy.columns = [(" " + c) if i % 2 else c for i, c in enumerate(messy.columns)]
result2 = nids.predict_flows(messy)
print("Same result as Test 1:", result1["predicted_class"].equals(result2["predicted_class"]))

# ---------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 3: a required column is missing")
print("=" * 70)
try:
    nids.predict_flows(X_sample.drop(columns=["Destination Port"]).head(100))
    print("No error raised (this is a problem)")
except ValueError as e:
    print("Raised ValueError:", e)

# ---------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 4: the raw Friday DDoS file (unclean values)")
print("=" * 70)
raw_files = sorted(RAW_DIR.glob("Friday-WorkingHours-Afternoon-DDos*.csv"))
if not raw_files:
    print("Raw file not found in", RAW_DIR, "- skipping")
else:
    start = time.time()
    raw = nids.read_flows(raw_files[0])
    result4 = nids.predict_flows(raw)
    print(f"File: {raw_files[0].name} | rows: {len(raw):,} | time: {time.time() - start:.0f}s")
    print("\nRow status:")
    print(result4["status"].value_counts().to_string())
    print("\nSummary:", nids.summarize(result4))

    if "Label" in raw.columns:
        scored = result4["status"] == nids.STATUS_OK
        truth = raw.loc[scored, "Label"].str.strip()
        pred = result4.loc[scored, "predicted_class"]
        print("\nTrue label (rows) vs predicted class (columns):")
        print(pd.crosstab(truth, pred))
        print("\nNOTE: this file was part of the training data, so these numbers only show that")
        print("the pipeline works on raw input. They are NOT a performance estimate.")

tmp_csv.unlink(missing_ok=True)