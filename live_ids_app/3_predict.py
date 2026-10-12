"""Step 3 (same machine as steps 1-2) - feed new flows to the trained model every 5 seconds
and print what the network looks like: Benign (normal) or which of the 14 attack types.

Model: new_isolationforest_xgboost_hybrid (stacked Isolation Forest + XGBoost, 95.85% offline
test accuracy - see new_isolationforest_xgboost_hybrid/training_report.pdf). Needs its 4 files
copied into ./models/ first - see README_RUN.txt step 3.

Strict rules this script follows:
  - Only the 63 features the model was trained on are read from the converter's CSV (via
    JAVA_COLUMN in app_common.py), cast to float32 (the dtype used at training).
  - A flow is used ONLY if every one of those 63 values is present and a finite number. Any
    flow missing a value, or with text/inf/NaN in a required column, is DROPPED and counted -
    never filled with 0, an average, or any other placeholder.
  - If a window has no usable flows, that is reported as-is ("no data"), never guessed.

Every result is also saved to disk, so you can check later what was normal and what was
abnormal:
  - results/flow_log.csv   one row per scored flow (time, source/destination, predicted
                            class, status = normal if Benign, abnormal otherwise, and
                            remote_host = a reverse-DNS name for the non-local IP, for
                            abnormal flows only - e.g. so "20.190.145.141" shows up as
                            an azureedge.net / githubusercontent.com style name, helping
                            you tell a real attack from ordinary HTTPS to a known service)
  - results/window_log.csv one row per 5-second window (counts, the window's verdict)
Both are opened in append mode and flushed after every write, so they are safe to read
while the app is still running, and nothing is lost if the window is closed.
The reverse-DNS lookup only runs for abnormal flows (to keep the normal-flow path fast),
is cached per IP, and is skipped (left blank) if it can't resolve within 0.3s - it never
blocks the 5-second cycle for long, and never invents a name.

Usage:  python3 3_predict.py
Stop with Ctrl+C.
"""
from app_common import CSV_DIR, JAVA_COLUMN, MODELS, RESULTS_DIR, WINDOW_SECONDS  # keep first (WMI patch)

import sys
sys.stdout.reconfigure(line_buffering=True, write_through=True)  # force real-time printing in every
sys.stderr.reconfigure(line_buffering=True, write_through=True)  # terminal host (PowerShell buffered this before)

import csv
import glob
import ipaddress
import json
import os
import socket
import time
from collections import Counter

import joblib
import numpy as np
import xgboost as xgb

_host_cache = {}


def is_private(ip):
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False  # not a valid IP (e.g. empty) - treat as not private so it isn't "resolved"


def remote_ip_of(row):
    """Which side of this flow is the non-local address (the one worth naming)."""
    src, dst = row.get("Src IP", ""), row.get("Dst IP", "")
    if src and not is_private(src):
        return src
    if dst and not is_private(dst):
        return dst
    return None


def resolve_host(ip):
    """Reverse-DNS name for ip, cached. Returns '' if it can't resolve - never guessed."""
    if ip is None:
        return ""
    if ip in _host_cache:
        return _host_cache[ip]
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(0.3)
    try:
        name = socket.gethostbyaddr(ip)[0]
    except (socket.herror, socket.gaierror, socket.timeout, OSError):
        name = ""
    finally:
        socket.setdefaulttimeout(old_timeout)
    _host_cache[ip] = name
    return name


def load_models():
    missing = [f for f in ("isolation_forest.joblib", "scaler.joblib", "xgboost_model.json", "if_config.json")
              if not os.path.exists(os.path.join(MODELS, f))]
    if missing:
        raise SystemExit(f"Missing model file(s) in {MODELS}: {missing}. "
                         f"Copy them from new_isolationforest_xgboost_hybrid/ first - see README_RUN.txt step 3.")
    cfg = json.load(open(os.path.join(MODELS, "if_config.json"), encoding="utf-8"))
    iforest = joblib.load(os.path.join(MODELS, "isolation_forest.joblib"))
    scaler = joblib.load(os.path.join(MODELS, "scaler.joblib"))
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODELS, "xgboost_model.json"))
    return cfg, iforest, scaler, model


def read_flows(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def build_matrix(rows, feats):
    """Returns (X float32 [n_usable, 63], kept_rows, n_dropped).
    A row is kept only if every required value is present and finite - no fakes, no blanks."""
    n = len(rows)
    X = np.full((n, len(feats)), np.nan, dtype=np.float64)
    for j, f in enumerate(feats):
        col = JAVA_COLUMN[f]
        for i, r in enumerate(rows):
            v = r.get(col)
            if v is None or v == "":
                continue  # stays NaN -> row dropped below, never filled
            try:
                X[i, j] = float(v)
            except ValueError:
                continue  # non-numeric text -> stays NaN -> row dropped
    ok = np.isfinite(X).all(axis=1)
    kept_rows = [r for r, k in zip(rows, ok) if k]
    return X[ok].astype(np.float32), kept_rows, int((~ok).sum())


def predict(X, iforest, scaler, model, classes):
    if len(X) == 0:
        return np.array([], dtype=object)
    score = -iforest.score_samples(scaler.transform(np.log1p(np.clip(X, 0, None) + 1.0)))
    X_stacked = np.hstack([X, score.reshape(-1, 1)]).astype(np.float32)
    idx = model.predict(X_stacked)
    return np.array(classes)[idx]


def open_logs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    flow_path = os.path.join(RESULTS_DIR, "flow_log.csv")
    window_path = os.path.join(RESULTS_DIR, "window_log.csv")
    flow_new = not os.path.exists(flow_path)
    window_new = not os.path.exists(window_path)
    flow_f = open(flow_path, "a", newline="", encoding="utf-8")
    window_f = open(window_path, "a", newline="", encoding="utf-8")
    flow_w = csv.writer(flow_f)
    window_w = csv.writer(window_f)
    if flow_new:
        flow_w.writerow(["time", "src_ip", "dst_ip", "dst_port", "predicted_label", "status", "remote_host"])
        flow_f.flush()
    if window_new:
        window_w.writerow(["time", "flows_captured", "flows_scored", "dropped", "verdict", "breakdown"])
        window_f.flush()
    print(f"logging every scored flow to {flow_path}", flush=True)
    print(f"logging every window's summary to {window_path}", flush=True)
    return flow_w, flow_f, window_w, window_f


def main():
    cfg, iforest, scaler, model = load_models()
    feats, classes = cfg["features_used"], cfg["classes"]
    benign_name = classes[cfg["benign_code"]]
    print(f"loaded model: {len(feats)} features + Isolation Forest score, {len(classes)} classes", flush=True)
    print(f"watching {CSV_DIR} every {WINDOW_SECONDS}s. Ctrl+C to stop.\n", flush=True)
    flow_w, flow_f, window_w, window_f = open_logs()

    seen = set()
    cumulative = Counter()
    cum_dropped = 0
    cum_total = 0

    try:
        while True:
            cycle_start = time.time()
            new_files = sorted(p for p in glob.glob(os.path.join(CSV_DIR, "*.csv")) if p not in seen)
            window_rows = []
            for p in new_files:
                seen.add(p)
                window_rows.extend(read_flows(p))

            stamp_full = time.strftime("%Y-%m-%d %H:%M:%S")
            stamp = time.strftime("%H:%M:%S")
            if not window_rows:
                print(f"[{stamp}] no new flows this window - NETWORK STATUS: (no data)", flush=True)
                window_w.writerow([stamp_full, 0, 0, 0, "(no data)", ""]); window_f.flush()
            else:
                X, kept_rows, dropped = build_matrix(window_rows, feats)
                pred_names = predict(X, iforest, scaler, model, classes)
                cum_dropped += dropped
                cum_total += len(pred_names)
                cumulative.update(pred_names.tolist())

                counts = Counter(pred_names.tolist())
                top_ips = Counter(r.get("Src IP", "?") for r in kept_rows).most_common(1)
                top_ip = top_ips[0][0] if top_ips else "?"

                abnormal_sources = []  # (label, ip, hostname) for the console summary below
                for r, name in zip(kept_rows, pred_names.tolist()):
                    status = "normal" if name == benign_name else "abnormal"
                    host = ""
                    if status == "abnormal":
                        ip = remote_ip_of(r)
                        host = resolve_host(ip)
                        abnormal_sources.append((name, ip or "?", host))
                    flow_w.writerow([stamp_full, r.get("Src IP", ""), r.get("Dst IP", ""),
                                     r.get("Dst Port", ""), name, status, host])
                flow_f.flush()

                print(f"[{stamp}] window: {len(window_rows)} flows captured, {len(pred_names)} scored, "
                      f"{dropped} dropped (missing/non-numeric value)", flush=True)
                if counts:
                    verdict = counts.most_common(1)[0][0]
                    breakdown = "; ".join(f"{n}={c}" for n, c in counts.most_common())
                    print(f"  NETWORK STATUS: {verdict}", flush=True)
                    for name, c in counts.most_common():
                        print(f"    {name:<28}{c:>5}  {c/len(pred_names)*100:6.1f}%", flush=True)
                    print(f"  most active source IP this window: {top_ip}", flush=True)
                    if abnormal_sources:
                        print(f"  abnormal flow sources this window:", flush=True)
                        for name, ip, host in abnormal_sources:
                            where = f"{ip} ({host})" if host else ip
                            print(f"    {name:<28}{where}", flush=True)
                else:
                    verdict = f"(no usable flows - all {dropped} incomplete)"
                    breakdown = ""
                    print(f"  NETWORK STATUS: {verdict}", flush=True)
                window_w.writerow([stamp_full, len(window_rows), len(pred_names), dropped, verdict, breakdown])
                window_f.flush()

            print(f"  cumulative: {cum_total} flows scored, {cum_dropped} dropped, since start", flush=True)
            print("-" * 70, flush=True)

            elapsed = time.time() - cycle_start
            time.sleep(max(0.0, WINDOW_SECONDS - elapsed))
    except KeyboardInterrupt:
        print("\nstopped.")
        print("final cumulative counts:")
        for name, c in cumulative.most_common():
            print(f"  {name:<28}{c:>6}")
    finally:
        flow_f.close()
        window_f.close()


if __name__ == "__main__":
    main()
