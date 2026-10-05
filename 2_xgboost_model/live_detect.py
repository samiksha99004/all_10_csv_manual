"""Live detection on a growing flow CSV (Kali converter output, Python cicflowmeter column names).

Each new row = one flow record. Every poll, new rows are aligned to the 78 model features
(names mapped explicitly, time columns converted from seconds to microseconds), scored by
level 1 (attack / not) and level 2 (which attack), and a per-second percentage table is printed.

Usage (run from this folder):
  python live_detect.py <flow.csv>            follow mode: re-read the file every 1 s, score new rows
  python live_detect.py <flow.csv> --replay   score the whole file at once (test run)
Console output only. Nothing is written to disk.
"""
import platform
platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())  # Windows WMI hangs imports

import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

m = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
FEATS, CLASSES, BENIGN = m["features"], m["classes"], m["benign_code"]
ATTACK_CLASSES = json.load(open(os.path.join(DATA, "attack_classes.json"), encoding="utf-8"))
A2C = np.array([CLASSES.index(ATTACK_CLASSES[str(i)]) for i in range(len(ATTACK_CLASSES))])

RENAME = {
    "Dst Port": "dst_port", "Protocol": "protocol", "Flow Duration": "flow_duration",
    "Tot Fwd Pkts": "tot_fwd_pkts", "Tot Bwd Pkts": "tot_bwd_pkts",
    "TotLen Fwd Pkts": "totlen_fwd_pkts", "TotLen Bwd Pkts": "totlen_bwd_pkts",
    "Fwd Pkt Len Max": "fwd_pkt_len_max", "Fwd Pkt Len Min": "fwd_pkt_len_min",
    "Fwd Pkt Len Mean": "fwd_pkt_len_mean", "Fwd Pkt Len Std": "fwd_pkt_len_std",
    "Bwd Pkt Len Max": "bwd_pkt_len_max", "Bwd Pkt Len Min": "bwd_pkt_len_min",
    "Bwd Pkt Len Mean": "bwd_pkt_len_mean", "Bwd Pkt Len Std": "bwd_pkt_len_std",
    "Flow Byts/s": "flow_byts_s", "Flow Pkts/s": "flow_pkts_s",
    "Flow IAT Mean": "flow_iat_mean", "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max", "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Tot": "fwd_iat_tot", "Fwd IAT Mean": "fwd_iat_mean", "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max", "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Tot": "bwd_iat_tot", "Bwd IAT Mean": "bwd_iat_mean", "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max", "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags", "Bwd PSH Flags": "bwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags", "Bwd URG Flags": "bwd_urg_flags",
    "Fwd Header Len": "fwd_header_len", "Bwd Header Len": "bwd_header_len",
    "Fwd Pkts/s": "fwd_pkts_s", "Bwd Pkts/s": "bwd_pkts_s",
    "Pkt Len Min": "pkt_len_min", "Pkt Len Max": "pkt_len_max", "Pkt Len Mean": "pkt_len_mean",
    "Pkt Len Std": "pkt_len_std", "Pkt Len Var": "pkt_len_var",
    "FIN Flag Cnt": "fin_flag_cnt", "SYN Flag Cnt": "syn_flag_cnt", "RST Flag Cnt": "rst_flag_cnt",
    "PSH Flag Cnt": "psh_flag_cnt", "ACK Flag Cnt": "ack_flag_cnt", "URG Flag Cnt": "urg_flag_cnt",
    "CWE Flag Count": "cwr_flag_count", "ECE Flag Cnt": "ece_flag_cnt",
    "Down/Up Ratio": "down_up_ratio", "Pkt Size Avg": "pkt_size_avg",
    "Fwd Seg Size Avg": "fwd_seg_size_avg", "Bwd Seg Size Avg": "bwd_seg_size_avg",
    "Fwd Byts/b Avg": "fwd_byts_b_avg", "Fwd Pkts/b Avg": "fwd_pkts_b_avg", "Fwd Blk Rate Avg": "fwd_blk_rate_avg",
    "Bwd Byts/b Avg": "bwd_byts_b_avg", "Bwd Pkts/b Avg": "bwd_pkts_b_avg", "Bwd Blk Rate Avg": "bwd_blk_rate_avg",
    "Subflow Fwd Pkts": "subflow_fwd_pkts", "Subflow Fwd Byts": "subflow_fwd_byts",
    "Subflow Bwd Pkts": "subflow_bwd_pkts", "Subflow Bwd Byts": "subflow_bwd_byts",
    "Init Fwd Win Byts": "init_fwd_win_byts", "Init Bwd Win Byts": "init_bwd_win_byts",
    "Fwd Act Data Pkts": "fwd_act_data_pkts", "Fwd Seg Size Min": "fwd_seg_size_min",
    "Active Mean": "active_mean", "Active Std": "active_std", "Active Max": "active_max", "Active Min": "active_min",
    "Idle Mean": "idle_mean", "Idle Std": "idle_std", "Idle Max": "idle_max", "Idle Min": "idle_min",
}
# Converter writes these in SECONDS; training uses MICROSECONDS.
TIME_FEATS = {
    "Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"}
TIME_SCALE = np.array([1e6 if f in TIME_FEATS else 1.0 for f in FEATS])
COL_IDX = [RENAME[f] for f in FEATS]

l1 = xgb.XGBClassifier(); l1.load_model(os.path.join(HERE, "level1_binary_xgb.json"))
l2 = xgb.XGBClassifier(); l2.load_model(os.path.join(HERE, "level2_multiclass_xgb.json"))


def read_rows(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def featurize(rows):
    """Return (X float32 for rows with all-finite features, list of kept rows)."""
    vals = np.full((len(rows), len(FEATS)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(COL_IDX):
            v = r.get(c, "")
            try:
                vals[i, j] = float(v)
            except ValueError:
                pass
    vals *= TIME_SCALE
    ok = np.isfinite(vals).all(axis=1)
    return vals[ok].astype(np.float32), [r for r, k in zip(rows, ok) if k], int((~ok).sum())


def score(X):
    """Level 1 then level 2, as in 4_evaluate.py. Returns (is_attack bool array, level-2 label names)."""
    is_att = l1.predict(X).astype(bool)
    names = np.array(["Benign"] * len(X), dtype=object)
    if is_att.any():
        names[is_att] = np.array(CLASSES)[A2C[l2.predict(X[is_att])]]
    return is_att, names


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    path = sys.argv[1]
    replay = "--replay" in sys.argv
    seen = 0
    tot = Counter(); tot_att = 0; tot_n = 0
    per_sec = defaultdict(Counter)  # second -> level-2 label counts for attack-flagged flows
    per_sec_n = Counter(); per_sec_att = Counter()

    while True:
        rows = read_rows(path)
        new = rows[seen:]
        if new:
            X, kept, dropped = featurize(new)
            if len(X):
                is_att, names = score(X)
                for r, a, nm in zip(kept, is_att, names):
                    sec = r["timestamp"]
                    per_sec_n[sec] += 1
                    if a:
                        per_sec_att[sec] += 1
                        per_sec[sec][nm] += 1
                    tot[nm] += 1
                tot_n += len(X); tot_att += int(is_att.sum())
            seen = len(rows)
            print(f"[{time.strftime('%H:%M:%S')}] +{len(new)} rows ({dropped} dropped inf/NaN) | total scored {tot_n}, "
                  f"attack-flagged {tot_att/max(tot_n,1)*100:.1f}%", flush=True)
        if replay:
            break
        time.sleep(1)

    print("\n" + "=" * 80)
    print("PER-SECOND TABLE (timestamp of each flow row; attack-flagged = level 1 'attack')")
    print("=" * 80)
    for sec in sorted(per_sec_n):
        n, a = per_sec_n[sec], per_sec_att[sec]
        top = ", ".join(f"{k} {v/a*100:.0f}%" for k, v in per_sec[sec].most_common(3)) if a else "-"
        print(f"{sec:<20} flows {n:>4}   attack {a/n*100:>5.1f}%   level-2: {top}")
    print("\nOVERALL (all scored rows, final label):")
    for k, v in tot.most_common():
        print(f"  {k:<28}{v:>6}  {v/tot_n*100:>6.2f}%")


if __name__ == "__main__":
    main()
