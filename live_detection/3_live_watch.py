"""Step 3 (Windows) - live watcher.

Watches a folder of flow CSVs written by 2_convert_loop.sh (Java CICFlowMeter-v3 format,
time in microseconds). For each NEW file it:
  1. maps the Java columns to the 78 training features (no unit scaling: Java already writes microseconds)
  2. scores the flows with the chosen model
  3. prints NORMAL or ABNORMAL, and for ABNORMAL a guess table of which attack

Usage (from the project root):
  python live_detection/3_live_watch.py                 # watch the live folder, model = 2level
  python live_detection/3_live_watch.py --model single  # model = single | hybrid | 2level
  python live_detection/3_live_watch.py --dir <folder> --once   # score the files already there, then exit

Console output only. Nothing is written to disk.
"""
import platform
platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())  # Windows WMI hangs imports

import csv
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import xgboost as xgb
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M1 = os.path.join(ROOT, "1_xgboost_model")
M2 = os.path.join(ROOT, "2_xgboost_model")
HY = os.path.join(ROOT, "isolation_forest_and_xgboost_model")
LIVE_DIR = os.path.join(ROOT, "live_detection", "live", "csv")
POLL_SECONDS = 2

meta = json.load(open(os.path.join(M2, "data", "meta.json"), encoding="utf-8"))
FEATS78, CLS78, BEN = meta["features"], meta["classes"], meta["benign_code"]
ac = json.load(open(os.path.join(M2, "data", "attack_classes.json"), encoding="utf-8"))
A2C = np.array([CLS78.index(ac[str(i)]) for i in range(len(ac))])

# Training feature -> Java CICFlowMeter-v3 header name (checked against the Java output file)
JAVA = {
    "Dst Port": "Dst Port", "Protocol": "Protocol", "Flow Duration": "Flow Duration",
    "Tot Fwd Pkts": "Total Fwd Packet", "Tot Bwd Pkts": "Total Bwd packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packet", "TotLen Bwd Pkts": "Total Length of Bwd Packet",
    "Fwd Pkt Len Max": "Fwd Packet Length Max", "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean", "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max", "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean", "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s", "Flow Pkts/s": "Flow Packets/s",
    "Flow IAT Mean": "Flow IAT Mean", "Flow IAT Std": "Flow IAT Std", "Flow IAT Max": "Flow IAT Max", "Flow IAT Min": "Flow IAT Min",
    "Fwd IAT Tot": "Fwd IAT Total", "Fwd IAT Mean": "Fwd IAT Mean", "Fwd IAT Std": "Fwd IAT Std", "Fwd IAT Max": "Fwd IAT Max", "Fwd IAT Min": "Fwd IAT Min",
    "Bwd IAT Tot": "Bwd IAT Total", "Bwd IAT Mean": "Bwd IAT Mean", "Bwd IAT Std": "Bwd IAT Std", "Bwd IAT Max": "Bwd IAT Max", "Bwd IAT Min": "Bwd IAT Min",
    "Fwd PSH Flags": "Fwd PSH Flags", "Bwd PSH Flags": "Bwd PSH Flags", "Fwd URG Flags": "Fwd URG Flags", "Bwd URG Flags": "Bwd URG Flags",
    "Fwd Header Len": "Fwd Header Length", "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s", "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Packet Length Min", "Pkt Len Max": "Packet Length Max", "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std", "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count", "SYN Flag Cnt": "SYN Flag Count", "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count", "ACK Flag Cnt": "ACK Flag Count", "URG Flag Cnt": "URG Flag Count",
    "CWE Flag Count": "CWR Flag Count", "ECE Flag Cnt": "ECE Flag Count", "Down/Up Ratio": "Down/Up Ratio",
    "Pkt Size Avg": "Average Packet Size", "Fwd Seg Size Avg": "Fwd Segment Size Avg", "Bwd Seg Size Avg": "Bwd Segment Size Avg",
    "Fwd Byts/b Avg": "Fwd Bytes/Bulk Avg", "Fwd Pkts/b Avg": "Fwd Packet/Bulk Avg", "Fwd Blk Rate Avg": "Fwd Bulk Rate Avg",
    "Bwd Byts/b Avg": "Bwd Bytes/Bulk Avg", "Bwd Pkts/b Avg": "Bwd Packet/Bulk Avg", "Bwd Blk Rate Avg": "Bwd Bulk Rate Avg",
    "Subflow Fwd Pkts": "Subflow Fwd Packets", "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets", "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "FWD Init Win Bytes", "Init Bwd Win Byts": "Bwd Init Win Bytes",
    "Fwd Act Data Pkts": "Fwd Act Data Pkts", "Fwd Seg Size Min": "Fwd Seg Size Min",
    "Active Mean": "Active Mean", "Active Std": "Active Std", "Active Max": "Active Max", "Active Min": "Active Min",
    "Idle Mean": "Idle Mean", "Idle Std": "Idle Std", "Idle Max": "Idle Max", "Idle Min": "Idle Min",
}
assert set(JAVA) == set(FEATS78)

# Models
l1 = xgb.XGBClassifier(); l1.load_model(os.path.join(M2, "level1_binary_xgb.json"))
l2 = xgb.XGBClassifier(); l2.load_model(os.path.join(M2, "level2_multiclass_xgb.json"))
mf1 = json.load(open(os.path.join(M1, "model_features.json")))["features"]
m1 = xgb.XGBClassifier(); m1.load_model(os.path.join(M1, "all_attacks_xgb.json"))
lc = {int(k): v for k, v in json.load(open(os.path.join(M1, "label_classes.json"))).items()}
cw = json.load(open(os.path.join(M1, "class_weights.json")))
W1 = np.array([cw[lc[c]] for c in range(len(lc))])
cfg = json.load(open(os.path.join(HY, "model_config.json")))
iforest = joblib.load(os.path.join(HY, "isolation_forest.joblib"))
scaler = joblib.load(os.path.join(HY, "scaler.joblib"))
s2 = xgb.Booster(); s2.load_model(os.path.join(HY, "xgboost_stage2.json"))


def load_rows(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def matrix(rows, train_names):
    X = np.empty((len(rows), len(train_names)), dtype=np.float64)
    for j, f in enumerate(train_names):
        for i, r in enumerate(rows):
            try:
                X[i, j] = float(r[JAVA[f]])
            except (KeyError, ValueError):
                X[i, j] = np.nan
    return X


def score(rows, model):
    """Returns (flagged bool array, guessed class name array) for the chosen model."""
    n = len(rows)
    flagged = np.zeros(n, dtype=bool)
    guess = np.array(["Benign"] * n, dtype=object)
    if n == 0:
        return flagged, guess
    if model == "2level":
        X = matrix(rows, FEATS78).astype(np.float32)
        att = l1.predict(X).astype(bool)
        flagged = att
        if att.any():
            guess[att] = np.array(CLS78)[A2C[l2.predict(X[att])]]
    elif model == "single":
        X = matrix(rows, mf1).astype(np.float32)
        P = m1.predict_proba(X)
        names = np.array([lc[c] for c in (P * W1).argmax(1)], dtype=object)
        guess = names
        flagged = names != "Benign"
    elif model == "hybrid":
        X = matrix(rows, cfg["features_used"]).astype(np.float32)
        s = -iforest.score_samples(scaler.transform(np.log1p(np.clip(X, 0, None) + 1)))
        att = s >= cfg["threshold"]
        flagged = att
        if att.any():
            guess[att] = np.array(cfg["classes"])[s2.predict(xgb.DMatrix(X[att])).argmax(1)]
    return flagged, guess


def report(name, rows, model):
    ok = np.isfinite(matrix(rows, FEATS78)).all(axis=1) if rows else np.array([], dtype=bool)
    rows = [r for r, k in zip(rows, ok) if k]
    flagged, guess = score(rows, model)
    n = len(rows)
    verdict = "ABNORMAL" if flagged.any() else "NORMAL"
    stamp = name.replace("seg_", "").split(".pcap")[0]
    print(f"\n[{time.strftime('%H:%M:%S')}] {stamp}  flows={n}  -> {verdict}")
    if verdict == "ABNORMAL":
        print(f"  flows flagged: {flagged.sum()} of {n} ({flagged.mean()*100:.1f}%)")
        print(f"  guess table (labels of flagged flows):")
        for lab, c in Counter(guess[flagged]).most_common():
            print(f"    {lab:<28}{c:>4}  {c/flagged.sum()*100:6.1f}%")
    sys.stdout.flush()


def main():
    args = sys.argv[1:]
    model = args[args.index("--model") + 1] if "--model" in args else "2level"
    folder = args[args.index("--dir") + 1] if "--dir" in args else LIVE_DIR
    once = "--once" in args
    os.makedirs(folder, exist_ok=True)
    print(f"watching {folder}  model={model}  poll={POLL_SECONDS}s", flush=True)
    seen = set()
    while True:
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".csv"))
        for f in files:
            if f in seen:
                continue
            seen.add(f)
            report(f, load_rows(os.path.join(folder, f)), model)
        if once:
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
