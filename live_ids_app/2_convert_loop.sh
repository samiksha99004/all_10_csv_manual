#!/bin/bash
# Step 2 (same machine as step 1) - converts each finished pcap with the Java CICFlowMeter-v3.
# Skips the newest pcap (tcpdump may still be writing it). Prunes old pcaps/csvs so disk usage
# stays bounded on a long-running capture.
#
# Usage:  bash 2_convert_loop.sh
# Needs: ~/live_ids_app/CICFlowMeter-4.0  (one-time copy of the built Java converter - see
# README_RUN.txt). If libpcap.so is not found by the converter, this project hit that before;
# the fix was a symlink - see README_RUN.txt step 4.
# Stop with Ctrl+C.

PCAP="$HOME/live_ids_app/pcap"
OUT="$HOME/live_ids_app/csv"
CONV="$HOME/live_ids_app/CICFlowMeter-4.0"
KEEP_FILES=40
MAX_FILES=60

mkdir -p "$OUT"
echo "converting new pcaps from $PCAP into $OUT (Ctrl+C to stop)"

while true; do
  newest=$(ls -t "$PCAP"/seg_*.pcap 2>/dev/null | head -1)
  for f in $(ls -tr "$PCAP"/seg_*.pcap 2>/dev/null); do
    [ "$f" = "$newest" ] && continue
    base=$(basename "$f")
    out_csv="$OUT/${base}_Flow.csv"
    [ -f "$out_csv" ] && continue
    (cd "$CONV" && java -Djava.library.path=lib/native -cp "lib/*" cic.cs.unb.ca.ifm.Cmd "$f" "$OUT" > /dev/null 2>&1)
    if [ -f "$out_csv" ]; then
      echo "converted $base -> $(basename "$out_csv")"
    else
      echo "WARNING: conversion did not produce $out_csv - check the Java converter is set up (README_RUN.txt step 4)"
    fi
  done

  # prune: keep the newest KEEP_FILES pcaps/csvs once more than MAX_FILES exist
  n_pcap=$(ls "$PCAP"/seg_*.pcap 2>/dev/null | wc -l)
  if [ "$n_pcap" -gt "$MAX_FILES" ]; then
    ls -tr "$PCAP"/seg_*.pcap | head -n $((n_pcap - KEEP_FILES)) | xargs -r rm -f
  fi
  n_csv=$(ls "$OUT"/*.csv 2>/dev/null | wc -l)
  if [ "$n_csv" -gt "$MAX_FILES" ]; then
    ls -tr "$OUT"/*.csv | head -n $((n_csv - KEEP_FILES)) | xargs -r rm -f
  fi

  sleep 1
done
