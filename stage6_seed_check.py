import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "all_cleaned.parquet"
MAP_FILE = BASE_DIR / "data" / "processed" / "label_map_model.json"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

N_FOLDS, GAP_FRAC, ATTACK_CAP, BENIGN_CAP = 5, 0.02, 100_000, 300_000
SEEDS = [0, 1, 2]
CONFIGS = [
    {"name": "leaf1_balanced", "leaf": 1,  "weight": "balanced_subsample"},
    {"name": "leaf1_sqrt",     "leaf": 1,  "weight": "sqrt"},
    {"name": "leaf5_sqrt",     "leaf": 5,  "weight": "sqrt"},
    {"name": "leaf10_sqrt",    "leaf": 10, "weight": "sqrt"},
    {"name": "leaf20_sqrt",    "leaf": 20, "weight": "sqrt"},
]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)

with open(MAP_FILE) as f:
    raw_map = json.load(f)
label_to_id = dict(raw_map) if all(isinstance(v, int) for v in raw_map.values()) \
    else {name: int(i) for i, name in raw_map.items()}
id_to_label = {i: n for n, i in label_to_id.items()}
class_ids = sorted(id_to_label)
benign_id = label_to_id["BENIGN"]
bpos = class_ids.index(benign_id)
bot_pos = class_ids.index(label_to_id["Bot"])

df = pd.read_parquet(DATA_FILE)
labels = df["Label"].astype(str).str.strip()
labels = labels.where(~labels.str.startswith("Web Attack"), "Web Attack")
keep = labels.isin(label_to_id)
feature_cols = [c for c in df.columns if c not in ("Label", "Target")]
X = df.loc[keep, feature_cols].to_numpy(dtype=np.float32)
y = labels[keep].map(label_to_id).to_numpy(dtype=np.int8)
del df, labels, keep
print("Rows used:", len(y), "| Features:", len(feature_cols))

block = np.zeros(len(y), dtype=np.int8)
rank = np.zeros(len(y), dtype=np.int64)
class_size = {}
for c in class_ids:
    idx = np.flatnonzero(y == c)
    r = np.arange(len(idx))
    rank[idx] = r
    block[idx] = r * N_FOLDS // len(idx)
    class_size[c] = len(idx)


def class_weight_for(spec, y_tr):
    if spec == "sqrt":
        counts = np.bincount(y_tr, minlength=len(class_ids)).astype(float)
        w = np.sqrt(len(y_tr) / (len(class_ids) * counts))
        return {int(c): float(w[c]) for c in class_ids}
    return spec


runs = {(cfg["name"], s): [] for cfg in CONFIGS for s in SEEDS}
t0 = time.time()
for seed in SEEDS:
    for fold in range(N_FOLDS):
        test_idx = np.flatnonzero(block == fold)
        parts = []
        for c in class_ids:
            entropy = [100 + fold, int(c)] + ([seed] if seed else [])
            rng = np.random.default_rng(entropy)
            in_c = (y == c)
            r_test = rank[in_c & (block == fold)]
            lo, hi = r_test.min(), r_test.max()
            gap = max(1, int(GAP_FRAC * class_size[c]))
            idx_c = np.flatnonzero(in_c & ((rank < lo - gap) | (rank > hi + gap)))
            cap = BENIGN_CAP if c == benign_id else ATTACK_CAP
            if len(idx_c) > cap:
                idx_c = rng.choice(idx_c, size=cap, replace=False)
            parts.append(idx_c)
        train_idx = np.concatenate(parts)
        X_tr, y_tr, X_te, y_te = X[train_idx], y[train_idx], X[test_idx], y[test_idx]

        for cfg in CONFIGS:
            model = RandomForestClassifier(
                n_estimators=100, n_jobs=-1, min_samples_leaf=cfg["leaf"],
                class_weight=class_weight_for(cfg["weight"], y_tr),
                random_state=42 + seed)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            cm = confusion_matrix(y_te, pred, labels=class_ids)
            n_benign = int(cm[bpos].sum())
            n_attack = int(cm.sum() - n_benign)
            false_alarms = n_benign - int(cm[bpos, bpos])
            missed = int(cm[:, bpos].sum() - cm[bpos, bpos])
            bot_tp = int(cm[bot_pos, bot_pos])
            runs[(cfg["name"], seed)].append({
                "macro_f1": f1_score(y_te, pred, average="macro", labels=class_ids, zero_division=0),
                "false_alarm_rate": false_alarms / n_benign,
                "attacks_caught": (n_attack - missed) / n_attack,
                "bot_recall": bot_tp / max(1, int(cm[bot_pos].sum())),
                "bot_precision": bot_tp / max(1, int(cm[:, bot_pos].sum())),
            })
        print(f"seed {seed} fold {fold + 1}/{N_FOLDS} done ({(time.time() - t0) / 60:.1f} min elapsed)")

per_seed = []
for (name, seed), folds in runs.items():
    f = pd.DataFrame(folds)
    per_seed.append({
        "config": name, "seed": seed,
        "macro_f1_mean": f["macro_f1"].mean(),
        "macro_f1_worst": f["macro_f1"].min(),
        "false_alarm_mean": f["false_alarm_rate"].mean(),
        "false_alarm_worst": f["false_alarm_rate"].max(),
        "attacks_caught": f["attacks_caught"].mean(),
        "bot_recall": f["bot_recall"].mean(),
        "bot_precision": f["bot_precision"].mean(),
    })
per_seed = pd.DataFrame(per_seed)
per_seed.to_csv(OUT_DIR / "stage6_seed_check_per_seed.csv", index=False)

print("\n" + "=" * 100)
print("PER SEED")
print(per_seed.round(4).to_string(index=False))

metrics = ["macro_f1_mean", "macro_f1_worst", "false_alarm_mean", "false_alarm_worst",
           "attacks_caught", "bot_recall", "bot_precision"]
rows = {}
for name, g in per_seed.groupby("config", sort=False):
    rows[name] = {}
    for m in metrics:
        if "false_alarm" in m:
            rows[name][m] = f"{g[m].mean():.3%} +/- {g[m].std(ddof=1):.3%}"
        else:
            rows[name][m] = f"{g[m].mean():.4f} +/- {g[m].std(ddof=1):.4f}"
summary = pd.DataFrame(rows)
print("\n" + "=" * 100)
print("MEAN +/- STD ACROSS 3 SEEDS")
print(summary)
summary.to_csv(OUT_DIR / "stage6_seed_check_summary.csv")
print("\nSaved to outputs/ | total time: %.1f min" % ((time.time() - t0) / 60))