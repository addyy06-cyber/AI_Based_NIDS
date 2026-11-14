import io
import secrets
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
import nids_predict as nids

REPORT_FILE = BASE_DIR / "outputs" / "report_per_class.csv"
MAX_UPLOAD_MB = 300
MAX_STORED_RESULTS = 2       # results live in memory only; the oldest one is forgotten
TIMELINE_BINS = 40
CONTEXT_COLS = ["Destination Port", "Flow Duration", "Total Fwd Packets",
                "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s"]
API_NAMES = {"destination_port": "Destination Port", "flow_duration": "Flow Duration",
             "fwd_packets": "Total Fwd Packets", "bwd_packets": "Total Backward Packets",
             "bytes_per_s": "Flow Bytes/s", "packets_per_s": "Flow Packets/s"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_, META = nids.load_model()          # load once at start-up; fails early if files are missing
RESULTS = OrderedDict()              # token -> results table (memory only, never saved to disk)


def reliability_level(precision):
    """High / Medium / Low, based on the precision measured in blocked cross-validation."""
    if precision >= 0.95:
        return "high"
    if precision >= 0.85:
        return "medium"
    return "low"


def load_reliability():
    if not REPORT_FILE.exists():
        return {}
    rel = pd.read_csv(REPORT_FILE, index_col=0)
    return {
        name: {
            "precision": round(float(row["pooled_precision"]), 3),
            "recall": round(float(row["recall_mean"]), 3),
            "recall_worst_fold": round(float(row["recall_worst_fold"]), 3),
            "level": reliability_level(float(row["pooled_precision"])),
        }
        for name, row in rel.iterrows()
    }


RELIABILITY = load_reliability()


def num(value):
    """Turn a table value into a JSON-safe number (None if missing)."""
    if pd.isna(value):
        return None
    value = float(value)
    return int(value) if value.is_integer() else round(value, 3)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"The file is larger than {MAX_UPLOAD_MB} MB."), 413


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model")
def api_model():
    return jsonify(
        trained_on=META["trained_on"],
        n_features=len(META["feature_columns"]),
        classes=list(META["classes"].values()),
        blocked_cv=META["blocked_cv_estimate_from_experiments"],
        reliability=RELIABILITY,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        return jsonify(error="No file was uploaded."), 400
    if not upload.filename.lower().endswith(".csv"):
        return jsonify(error="Please upload a .csv file."), 400

    start = time.time()
    try:
        flows = nids.read_flows(io.BytesIO(upload.read()))
        if len(flows) == 0:
            return jsonify(error="The file contains no rows."), 400
        result = nids.predict_flows(flows)
    except ValueError as e:                     # for example: required columns are missing
        return jsonify(error=str(e)), 400
    except Exception:
        app.logger.exception("Could not process the uploaded file")
        return jsonify(error="Could not read this file as a CSV of network flows."), 400

    context = flows[[c for c in CONTEXT_COLS if c in flows.columns]].apply(
        pd.to_numeric, errors="coerce")
    table = pd.concat([result, context], axis=1)
    table.insert(0, "row", np.arange(1, len(table) + 1))

    token = secrets.token_urlsafe(16)
    RESULTS[token] = table
    while len(RESULTS) > MAX_STORED_RESULTS:
        RESULTS.popitem(last=False)

    summary = nids.summarize(result)
    scored_mask = (table["status"] == nids.STATUS_OK).to_numpy()
    attack_mask = table["is_attack"].fillna(False).astype(bool).to_numpy()
    counts = table.loc[scored_mask, "predicted_class"].value_counts()
    attack_counts = counts.drop("BENIGN", errors="ignore")

    # "Traffic overview": flows per position in the file (row order ~ capture order)
    n = len(table)
    bins = np.arange(n) * TIMELINE_BINS // n
    benign_per_bin = np.bincount(bins[scored_mask & ~attack_mask], minlength=TIMELINE_BINS)
    attack_per_bin = np.bincount(bins[attack_mask], minlength=TIMELINE_BINS)

    return jsonify(
        token=token,
        seconds=round(time.time() - start, 1),
        rows_uploaded=summary["rows_uploaded"],
        rows_scored=summary["rows_scored"],
        rows_not_scored=summary["rows_not_scored"],
        not_scored_reasons={k: int(v) for k, v in summary["not_scored_reasons"].items()},
        attacks_flagged=summary["attacks_flagged"],
        alert_share=round(summary["attacks_flagged"] / max(summary["rows_scored"], 1), 4),
        class_counts={k: int(v) for k, v in counts.items()},
        reliability={k: RELIABILITY[k] for k in attack_counts.index if k in RELIABILITY},
        timeline={
            "bin_start_percent": [round(100 * i / TIMELINE_BINS, 1) for i in range(TIMELINE_BINS)],
            "normal": [int(v) for v in benign_per_bin],
            "attack": [int(v) for v in attack_per_bin],
        },
    )


@app.route("/api/alerts")
def api_alerts():
    table = RESULTS.get(request.args.get("token", ""))
    if table is None:
        return jsonify(error="These results are no longer available. Please upload the file again."), 404

    scored = table[table["status"] == nids.STATUS_OK]
    alerts = scored[scored["is_attack"].astype(bool)]

    wanted = [c for c in request.args.get("classes", "").split(",") if c]
    if wanted:
        alerts = alerts[alerts["predicted_class"].isin(wanted)]

    try:
        min_conf = min(max(float(request.args.get("min_conf", 0)), 0.0), 1.0)
        page = max(int(request.args.get("page", 1)), 1)
        per_page = min(max(int(request.args.get("per_page", 25)), 5), 100)
    except ValueError:
        return jsonify(error="Invalid filter value."), 400
    alerts = alerts[alerts["confidence"] >= min_conf]

    query = request.args.get("q", "").strip()[:50]
    if query:
        ports = alerts["Destination Port"].astype("int64").astype(str)
        by_class = alerts["predicted_class"].str.contains(query, case=False, regex=False)
        alerts = alerts[by_class | ports.str.startswith(query)]

    alerts = alerts.sort_values("attack_score", ascending=False)
    page_df = alerts.iloc[(page - 1) * per_page: page * per_page]

    rows = []
    for _, r in page_df.iterrows():
        item = {
            "row": int(r["row"]),
            "predicted_class": r["predicted_class"],
            "confidence": round(float(r["confidence"]), 3),
            "attack_score": round(float(r["attack_score"]), 3),
            "reliability": RELIABILITY.get(r["predicted_class"], {}).get("level", "unknown"),
        }
        for api_name, column in API_NAMES.items():
            item[api_name] = num(r[column]) if column in page_df.columns else None
        rows.append(item)

    return jsonify(total=int(len(alerts)), page=page, per_page=per_page, rows=rows)


@app.route("/api/download/<token>/<kind>")
def api_download(token, kind):
    table = RESULTS.get(token)
    if table is None or kind not in ("all", "alerts"):
        return jsonify(error="Not found. Please upload the file again."), 404
    if kind == "alerts":
        is_alert = table["is_attack"].fillna(False).astype(bool)
        table = table[is_alert].sort_values("attack_score", ascending=False)
    return Response(table.to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=nids_{kind}.csv"})


if __name__ == "__main__":
    # 127.0.0.1 = this computer only. Do not change it to 0.0.0.0. Debug stays off on purpose.
    app.run(host="127.0.0.1", port=8000, debug=False)