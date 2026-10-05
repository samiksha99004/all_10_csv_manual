#!/bin/bash
# Step 2 (ATTACKER Kali) - convert each finished 2-second pcap into a flow CSV.
# Reads pcaps from the shared folder written by 1_capture_rotate.sh, writes CSVs next to them
# in live/csv (which the Windows watcher reads). Converts only new files and skips the newest
# pcap, because tcpdump may still be writing it.
# Usage:  bash 2_convert_loop.sh      (Ctrl+C to stop)

PCAP=/media/sf_F_DRIVE/all_10_csv_manual/live_detection/live/pcap
OUT=/media/sf_F_DRIVE/all_10_csv_manual/live_detection/live/csv
CONV=/home/kali/CICFlowMeter/build/distributions/CICFlowMeter-4.0

mkdir -p "$OUT" || { echo "cannot write $OUT - is the share mounted?"; exit 1; }
echo "converting new pcaps from $PCAP into $OUT (Ctrl+C to stop)"

while true; do
  newest=$(ls -t "$PCAP"/seg_*.pcap 2>/dev/null | head -1)
  for f in $(ls -tr "$PCAP"/seg_*.pcap 2>/dev/null); do
    [ "$f" = "$newest" ] && continue
    base=$(basename "$f")
    [ -f "$OUT/${base}_Flow.csv" ] && continue
    (cd "$CONV" && java -Djava.library.path=lib/native -cp "lib/*" cic.cs.unb.ca.ifm.Cmd "$f" "$OUT" > /dev/null 2>&1)
    echo "converted $base -> ${base}_Flow.csv"
  done
  sleep 1
done
