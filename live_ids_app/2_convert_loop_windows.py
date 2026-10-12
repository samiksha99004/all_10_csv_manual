"""Step 2 (Windows) - converts each finished pcap with the Java CICFlowMeter-v3.
Skips the newest pcap (the capture script may still be writing it). Prunes old pcaps/csvs so
disk usage stays bounded on a long-running capture.

Usage:  python 2_convert_loop_windows.py
Needs CONV_DIR (app_common.py) to point at a built CICFlowMeter-4.0 - already set to this
machine's F:\\CICFlowMeter-master\\build\\distributions\\CICFlowMeter-4.0.
Stop with Ctrl+C.
"""
from app_common import CONV_DIR, CSV_DIR, KEEP_FILES, MAX_FILES, PCAP_DIR  # keep first (WMI patch)

import sys
sys.stdout.reconfigure(line_buffering=True, write_through=True)  # force real-time printing
sys.stderr.reconfigure(line_buffering=True, write_through=True)

import glob
import os
import subprocess
import time


def convert(pcap_path):
    out_csv = os.path.join(CSV_DIR, os.path.basename(pcap_path) + "_Flow.csv")
    if os.path.exists(out_csv):
        return out_csv
    subprocess.run(
        ["java", f"-Djava.library.path=lib/native", "-cp", "lib/*", "cic.cs.unb.ca.ifm.Cmd", pcap_path, CSV_DIR],
        cwd=CONV_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_csv if os.path.exists(out_csv) else None


def prune(folder, pattern):
    files = sorted(glob.glob(os.path.join(folder, pattern)), key=os.path.getmtime)
    if len(files) > MAX_FILES:
        for f in files[:len(files) - KEEP_FILES]:
            os.remove(f)


def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    if not os.path.isdir(CONV_DIR):
        raise SystemExit(f"CICFlowMeter not found at {CONV_DIR} - check app_common.py CONV_DIR.")
    print(f"converting new pcaps from {PCAP_DIR} into {CSV_DIR} (Ctrl+C to stop)", flush=True)

    done = set()
    try:
        while True:
            pcaps = sorted(glob.glob(os.path.join(PCAP_DIR, "seg_*.pcap")), key=os.path.getmtime)
            newest = pcaps[-1] if pcaps else None
            for p in pcaps:
                if p == newest or p in done:
                    continue
                out = convert(p)
                done.add(p)
                stamp = time.strftime("%H:%M:%S")
                if out:
                    print(f"[{stamp}] converted {os.path.basename(p)} -> {os.path.basename(out)}", flush=True)
                else:
                    print(f"[{stamp}] WARNING: conversion failed for {os.path.basename(p)}", flush=True)
            prune(PCAP_DIR, "seg_*.pcap")
            prune(CSV_DIR, "*.csv")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
