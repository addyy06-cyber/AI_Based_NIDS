import threading
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

BASE_DIR = Path(__file__).resolve().parent.parent
DEST = BASE_DIR / "data" / "external_2018"
DEST.mkdir(parents=True, exist_ok=True)

BUCKET = "cse-cic-ids2018"
PREFIX = "Processed Traffic Data for ML Algorithms/"
REGION = "ca-central-1"
FILES = [
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
]

s3 = boto3.client("s3", region_name=REGION, config=Config(signature_version=UNSIGNED))


class Progress:
    """Prints a line every 25% of a download."""
    def __init__(self, name, total):
        self.name, self.total, self.done, self.next_mark = name, total, 0, 25
        self.lock = threading.Lock()

    def __call__(self, nbytes):
        with self.lock:
            self.done += nbytes
            while self.total and 100 * self.done / self.total >= self.next_mark <= 100:
                print(f"    {self.name}: {self.next_mark}%")
                self.next_mark += 25


# ---------------------------------------------------------
# 1. Download (skips files that are already complete)
# ---------------------------------------------------------
print("DOWNLOAD")
for name in FILES:
    key = PREFIX + name
    remote_size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    path = DEST / name
    if path.exists() and path.stat().st_size == remote_size:
        print(f"  {name}: already downloaded ({remote_size / 1e6:,.0f} MB)")
        continue
    print(f"  {name}: downloading {remote_size / 1e6:,.0f} MB ...")
    s3.download_file(BUCKET, key, str(path), Callback=Progress(name, remote_size))
    ok = path.stat().st_size == remote_size
    print(f"  {name}: {'OK, size matches the server' if ok else 'SIZE MISMATCH, run the script again'}")

# ---------------------------------------------------------
# 2. Inspect the header and the labels of each file
# ---------------------------------------------------------
print("\nINSPECTION")
reference_columns = None
for name in FILES:
    path = DEST / name
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if reference_columns is None:
        reference_columns = columns
        print(f"\nColumns of the first file ({len(columns)}):")
        print(columns)

    label_frame = pd.read_csv(path, usecols=lambda c: c.strip().lower() == "label",
                              dtype=str, low_memory=False)
    labels = label_frame.iloc[:, 0].str.strip()
    repeated_headers = int((labels.str.lower() == "label").sum())

    print(f"\n{name}")
    print(f"  rows: {len(labels):,} | same columns as the first file: {columns == reference_columns}"
          f" | repeated header rows inside the data: {repeated_headers}")
    print(labels.value_counts().to_string())