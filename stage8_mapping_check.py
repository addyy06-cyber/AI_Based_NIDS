import json
from pathlib import Path

import numpy as np
import pandas as pd

from cic2018_mapping import FEATURE_MAP, LABEL_MAP

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "external_2018"
FILES = sorted(DATA_DIR.glob("*_TrafficForML_CICFlowMeter.csv"))
REFERENCE_FILE = DATA_DIR / "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"   # almost all benign

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

with open(BASE_DIR / "outputs" / "final_model_meta.json") as f:
    features = json.load(f)["feature_columns"]

# ---------------------------------------------------------
print("=" * 70)
print("1. FEATURE MAPPING")
print("=" * 70)
print("Model features:", len(features), "| entries in the mapping:", len(FEATURE_MAP))
print("Model features without a mapping:", [f for f in features if f not in FEATURE_MAP])
print("Mapping entries the model does not use:", [f for f in FEATURE_MAP if f not in features])
header = pd.read_csv(REFERENCE_FILE, nrows=0).columns.str.strip().tolist()
print("Mapped 2018 columns missing from the file:", [c for c in FEATURE_MAP.values() if c not in header])

# ---------------------------------------------------------
print("\n" + "=" * 70)
print("2. LABEL MAPPING (row counts before any cleaning)")
print("=" * 70)
totals, unmapped = {}, {}
for path in FILES:
    labels = pd.read_csv(path, usecols=lambda c: c.strip().lower() == "label",
                         dtype=str, low_memory=False).iloc[:, 0].str.strip()
    for name, n in labels.value_counts().items():
        if name in LABEL_MAP:
            totals[LABEL_MAP[name]] = totals.get(LABEL_MAP[name], 0) + int(n)
        else:
            unmapped[name] = unmapped.get(name, 0) + int(n)
print(pd.Series(totals).sort_values(ascending=False).to_string())
print("Labels with no mapping (ignored):", unmapped)

# ---------------------------------------------------------
print("\n" + "=" * 70)
print("3. DO THE FEATURES MEAN THE SAME THING? (benign flows only, 200,000 from each dataset)")
print("=" * 70)
ref = pd.read_parquet(BASE_DIR / "data" / "processed" / "all_cleaned.parquet",
                      columns=features + ["Label"])
ref = ref[ref["Label"].astype(str).str.strip() == "BENIGN"][features]
ref = ref.sample(200_000, random_state=0).astype("float64")

new = pd.read_csv(REFERENCE_FILE, low_memory=False)
new.columns = new.columns.str.strip()
new = new[new["Label"].str.strip() == "Benign"]
new = new[list(FEATURE_MAP.values())].apply(pd.to_numeric, errors="coerce")
new = new.rename(columns={v: k for k, v in FEATURE_MAP.items()})[features]
new = new.sample(200_000, random_state=0)


def summarize(frame):
    clean = frame.replace([np.inf, -np.inf], np.nan)
    return clean.median(), clean.quantile(0.95)


ref_med, ref_p95 = summarize(ref)
new_med, new_p95 = summarize(new)
table = pd.DataFrame({"CICIDS2017_median": ref_med, "2018_median": new_med,
                      "CICIDS2017_p95": ref_p95, "2018_p95": new_p95})


def far_apart(a, b):
    """True if two values differ by more than 5x, or one is positive and the other is not."""
    both_positive = (a > 0) & (b > 0)
    ratio = (b / a).where(both_positive)
    return (ratio > 5) | (ratio < 0.2) | ((a > 0) != (b > 0))


flag = far_apart(table["CICIDS2017_median"], table["2018_median"]) | \
       far_apart(table["CICIDS2017_p95"], table["2018_p95"])

print("\nA few features to check for unit changes (always shown):")
key = ["Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
       "Flow Bytes/s", "Flow IAT Mean", "Idle Mean", "Init_Win_bytes_forward", "Destination Port"]
print(table.loc[key].round(2).to_string())

print(f"\nFeatures whose typical values differ by more than 5x (or zero vs non-zero): "
      f"{int(flag.sum())} of {len(features)}")
print(table[flag].round(2).to_string())