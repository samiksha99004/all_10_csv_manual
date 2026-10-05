"""Live SSH brute-force monitor - runs ON THE VICTIM.

The victim does everything:
  1. tcpdump (run separately, see README) writes a new pcap every 2 seconds into PCAP_DIR
  2. this script converts each finished pcap with the Java CICFlowMeter-v3 (time in microseconds)
  3. selects the 78 training features by name
  4. scores the flows with the 2-level XGBoost (level 1 attack/benign, level 2 which attack)
  5. redraws a terminal table every few seconds: attacker IP, NORMAL/ABNORMAL, guess table

Usage:
  python3 live_monitor.py                       # default folders (see the constants below)
  python3 live_monitor.py --refresh 3           # redraw every 3 seconds

Ctrl+C to stop. Nothing is written except the converter's flow CSVs (in CSV_DIR).
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
HOME = os.path.expanduser("~")
PCAP_DIR = os.path.join(HOME, "ids", "pcap")
CSV_DIR = os.path.join(HOME, "ids", "csv")
CONV_DIR = os.path.join(HOME, "ids", "CICFlowMeter-4.0")

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


def load_models():
    meta = json.load(open(os.path.join(MODELS, "meta.json"), encoding="utf-8"))
    ac = json.load(open(os.path.join(MODELS, "attack_classes.json"), encoding="utf-8"))
    classes = meta["classes"]
    a2c = np.array([classes.index(ac[str(i)]) for i in range(len(ac))])
    l1 = xgb.XGBClassifier(); l1.load_model(os.path.join(MODELS, "level1_binary_xgb.json"))
    l2 = xgb.XGBClassifier(); l2.load_model(os.path.join(MODELS, "level2_multiclass_xgb.json"))
    return meta["features"], classes, a2c, l1, l2


def convert(pcap, out_dir):
    """Java CICFlowMeter-v3 on one pcap. Returns the CSV path it wrote, or None."""
    subprocess.run(["java", "-Djava.library.path=lib/native", "-cp", "lib/*",
                    "cic.cs.unb.ca.ifm.Cmd", pcap, out_dir],
                   cwd=CONV_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = os.path.join(out_dir, os.path.basename(pcap) + "_Flow.csv")
    return out if os.path.exists(out) else None


def read_flows(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def score_rows(rows, feats, classes, a2c, l1, l2):
    """Returns (rows kept, flagged bool array, guessed class name array)."""
    if not rows:
        return [], np.array([], bool), np.array([], object)
    X = np.empty((len(rows), len(feats)), dtype=np.float64)
    for j, f in enumerate(feats):
        for i, r in enumerate(rows):
            try:
                X[i, j] = float(r[JAVA[f]])
            except (KeyError, ValueError):
                X[i, j] = np.nan
    ok = np.isfinite(X).all(axis=1)
    rows = [r for r, k in zip(rows, ok) if k]
    X = X[ok].astype(np.float32)
    flagged = l1.predict(X).astype(bool) if len(X) else np.array([], bool)
    guess = np.array(["Benign"] * len(X), dtype=object)
    if flagged.any():
        guess[flagged] = np.array(classes)[a2c[l2.predict(X[flagged])]]
    return rows, flagged, guess


def attacker_guess(rows_all):
    """Attacker = the source IP that sends the most flows to port 22."""
    c = Counter(r["Src IP"] for r in rows_all if r.get("Dst Port") == "22")
    return c.most_common(1)[0][0] if c else "none seen yet"


def render(state):
    out = []
    out.append("=" * 72)
    out.append("  SSH BRUTE-FORCE LIVE MONITOR   (victim: this machine)")
    out.append("=" * 72)
    out.append(f"  time now         : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(f"  attacker IP      : {state['attacker']}")
    out.append(f"  segments done    : {state['segments']}   flows seen: {state['flows_total']}")
    out.append("")
    v = state["verdict"]
    out.append(f"  CURRENT WINDOW   : {v}   ({state['win_flows']} flows, {state['win_flagged']} flagged)")
    if state["win_guess"]:
        out.append("  guess table (this window, flagged flows):")
        for lab, c in state["win_guess"]:
            out.append(f"    {lab:<26}{c:>5}   {c / state['win_flagged'] * 100:6.1f}%")
    out.append("")
    out.append("  CUMULATIVE (all windows so far):")
    out.append(f"    normal (not flagged) : {state['cum_normal']:>6}  {state['cum_normal_pct']:6.1f}%")
    out.append(f"    flagged as attack    : {state['cum_flagged']:>6}  {state['cum_flagged_pct']:6.1f}%")
    if state["cum_guess"]:
        out.append("  guess table (cumulative, flagged flows):")
        for lab, c in state["cum_guess"]:
            out.append(f"    {lab:<26}{c:>6}   {c / max(state['cum_flagged'], 1) * 100:6.1f}%")
    out.append("")
    out.append("  Ctrl+C to stop")
    return "\n".join(out)


def main():
    global PCAP_DIR, CSV_DIR, CONV_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", type=float, default=2.0, help="seconds between screen updates")
    ap.add_argument("--pcap-dir", default=PCAP_DIR)
    ap.add_argument("--csv-dir", default=CSV_DIR)
    ap.add_argument("--conv-dir", default=CONV_DIR)
    ap.add_argument("--csv-only", action="store_true", help="skip conversion; read CSVs already in csv-dir (testing)")
    args = ap.parse_args()
    PCAP_DIR, CSV_DIR, CONV_DIR = args.pcap_dir, args.csv_dir, args.conv_dir
    os.makedirs(CSV_DIR, exist_ok=True)

    feats, classes, a2c, l1, l2 = load_models()
    done = set()
    state = {"attacker": "none seen yet", "segments": 0, "flows_total": 0,
             "verdict": "WAITING", "win_flows": 0, "win_flagged": 0, "win_guess": [],
             "cum_normal": 0, "cum_flagged": 0, "cum_normal_pct": 0.0, "cum_flagged_pct": 0.0, "cum_guess": []}
    cum_guess = Counter()
    all_rows = []
    try:
        while True:
            if args.csv_only:
                todo = [p for p in sorted(glob.glob(os.path.join(CSV_DIR, "*.csv"))) if p not in done]
                csv_files = todo
            else:
                pcaps = sorted(glob.glob(os.path.join(PCAP_DIR, "seg_*.pcap")))
                newest = pcaps[-1] if pcaps else None
                csv_files = []
                for p in pcaps:
                    if p == newest or p in done:
                        continue
                    out = convert(p, CSV_DIR)
                    done.add(p)
                    if out:
                        csv_files.append(out)
            for path in csv_files:
                if args.csv_only:
                    done.add(path)
                rows = read_flows(path)
                kept, flagged, guess = score_rows(rows, feats, classes, a2c, l1, l2)
                all_rows.extend(kept)
                state["segments"] += 1
                state["flows_total"] += len(kept)
                state["win_flows"] = len(kept)
                state["win_flagged"] = int(flagged.sum())
                state["verdict"] = "ABNORMAL" if flagged.any() else "NORMAL"
                state["win_guess"] = Counter(guess[flagged]).most_common() if flagged.any() else []
                cum_guess.update(guess[flagged].tolist())
                cum_flagged = sum(cum_guess.values())
                state["cum_flagged"] = cum_flagged
                state["cum_normal"] = state["flows_total"] - cum_flagged
                tot = max(state["flows_total"], 1)
                state["cum_flagged_pct"] = cum_flagged / tot * 100
                state["cum_normal_pct"] = state["cum_normal"] / tot * 100
                state["cum_guess"] = cum_guess.most_common()
            state["attacker"] = attacker_guess(all_rows)
            print("\033[2J\033[H" + render(state), flush=True)
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
