import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import nids_predict as nids
from cic2018_mapping import FEATURE_MAP, LABEL_MAP

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "external_2018"
FILES = sorted(DATA_DIR.glob("*_TrafficForML_CICFlowMeter.csv"))

BENIGN_PER_FILE = 25_000          # benign flows sampled from each 2018 file
ATTACK_PER_FILE_CLASS = 10_000    # flows per attack class sampled from each 2018 file
OLD_BENIGN = 150_000
OLD_ATTACK = 50_000
CLASSES_TO_COMPARE = ["DDoS", "DoS Hulk", "Bot", "SSH-Patator"]

pd.set_option("display.width", 220)
pd.set_option("display.float_format", "{:,.2f}".format)

model, meta = nids.load_model()
features = meta["feature_columns"]
RENAME = {v: k for k, v in FEATURE_MAP.items()}
WANTED = set(FEATURE_MAP.values()) | {"Label"}
rng = np.random.default_rng(0)


def load_file(path):
    xs, ys = [], []
    for chunk in pd.read_csv(path, usecols=lambda c: c.strip() in WANTED, chunksize=100_000):
        chunk.columns = chunk.columns.str.strip()
        label = chunk["Label"].astype(str).str.strip()
        known = label.isin(LABEL_MAP)
        chunk, label = chunk[known], label[known]
        x = chunk[list(FEATURE_MAP.values())].apply(pd.to_numeric, errors="coerce")
        xs.append(x.rename(columns=RENAME)[features].astype(np.float32))
        ys.append(label.map(LABEL_MAP))
    return pd.concat(xs, ignore_index=True), pd.concat(ys, ignore_index=True)


# ---------------------------------------------------------
# 1. Sample flows from the 2018 files (valid rows only)
# ---------------------------------------------------------
samples = []
for path in FILES:
    X, y = load_file(path)
    X = X.replace([np.inf, -np.inf], np.nan)
    bad = X.isna().any(axis=1) | (X[nids.NEGATIVE_TIMING_COLS] < 0).any(axis=1)
    X, y = X[~bad].reset_index(drop=True), y[~bad.to_numpy()].reset_index(drop=True)
    labels = y.to_numpy()
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        n = BENIGN_PER_FILE if cls == "BENIGN" else ATTACK_PER_FILE_CLASS
        if len(idx) > n:
            idx = rng.choice(idx, size=n, replace=False)
        part = X.iloc[idx].astype("float64")
        part["cls"] = cls
        samples.append(part)
    del X, y
    gc.collect()
new = pd.concat(samples, ignore_index=True)
del samples
print("2018 sample sizes:", new["cls"].value_counts().to_dict())

# ---------------------------------------------------------
# 2. Sample flows from CICIDS2017
# ---------------------------------------------------------
ref = pd.read_parquet(BASE_DIR / "data" / "processed" / "all_cleaned.parquet", columns=features + ["Label"])
labels17 = ref["Label"].astype(str).str.strip()
labels17 = labels17.where(~labels17.str.startswith("Web Attack"), "Web Attack").to_numpy()
ref = ref[features]
parts = []
for cls in ["BENIGN"] + CLASSES_TO_COMPARE:
    idx = np.flatnonzero(labels17 == cls)
    n = OLD_BENIGN if cls == "BENIGN" else OLD_ATTACK
    if len(idx) > n:
        idx = rng.choice(idx, size=n, replace=False)
    part = ref.iloc[idx].astype("float64")
    part["cls"] = cls
    parts.append(part)
old = pd.concat(parts, ignore_index=True)
del ref, parts, labels17
gc.collect()
print("2017 sample sizes:", old["cls"].value_counts().to_dict())

# ---------------------------------------------------------
# 3. Diagnostic 1: can a classifier tell the two datasets apart using BENIGN flows only?
# ---------------------------------------------------------
print("\n" + "=" * 90)
print("DIAGNOSTIC 1: can a classifier tell 2017 benign from 2018 benign?")
print("=" * 90)
new_b = new[new["cls"] == "BENIGN"][features]
old_b = old[old["cls"] == "BENIGN"][features]
n = min(len(new_b), len(old_b))
new_b, old_b = new_b.sample(n, random_state=0), old_b.sample(n, random_state=0)
X_adv = pd.concat([old_b, new_b], ignore_index=True)
y_adv = np.r_[np.zeros(n), np.ones(n)]
X_tr, X_te, y_tr, y_te = train_test_split(X_adv, y_adv, test_size=0.25, stratify=y_adv, random_state=0)


def fit_and_score(columns):
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=0)
    clf.fit(X_tr[columns], y_tr)
    prob = clf.predict_proba(X_te[columns])[:, 1]
    return clf, roc_auc_score(y_te, prob), ((prob > 0.5) == y_te).mean()


clf, auc, acc = fit_and_score(features)
print(f"Using all {len(features)} features: AUC {auc:.4f}, accuracy {acc:.4f}   (0.5 = cannot tell apart, 1.0 = perfectly)")
importance = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeatures that best separate the two datasets (benign traffic only):")
print(importance.head(15).round(4).to_string())

remaining = [f for f in features if f not in importance.head(5).index]
_, auc2, acc2 = fit_and_score(remaining)
print(f"\nWithout the 5 most important features: AUC {auc2:.4f}, accuracy {acc2:.4f}")

# ---------------------------------------------------------
# 4. Diagnostic 2: do 2018 attacks look like 2017 attacks or like 2017 benign traffic?
# ---------------------------------------------------------
print("\n" + "=" * 90)
print("DIAGNOSTIC 2: typical (median) values of the model's 8 most important features")
print("=" * 90)
top_features = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(8).index.tolist()


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


b17 = old.loc[old["cls"] == "BENIGN", top_features].median()
b18 = new.loc[new["cls"] == "BENIGN", top_features].median()
for cls in CLASSES_TO_COMPARE:
    a17 = old.loc[old["cls"] == cls, top_features].median()
    a18 = new.loc[new["cls"] == cls, top_features].median()
    table = pd.DataFrame({f"2017 {cls}": a17, f"2018 {cls}": a18, "2017 BENIGN": b17, "2018 BENIGN": b18})
    d_attack = (slog(a18) - slog(a17)).abs()
    d_benign = (slog(a18) - slog(b17)).abs()
    table["2018 attack closer to"] = np.where(d_attack < d_benign, "2017 attack",
                                     np.where(d_attack > d_benign, "2017 benign", "tie"))
    print(f"\n--- {cls} ---")
    print(table.to_string())
    n_attack = int((table["2018 attack closer to"] == "2017 attack").sum())
    n_benign = int((table["2018 attack closer to"] == "2017 benign").sum())
    print(f"-> closer to the 2017 attack on {n_attack} of {len(top_features)} features, "
          f"closer to 2017 benign on {n_benign}")