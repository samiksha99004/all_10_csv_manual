#!/bin/bash
# Step 1 (VICTIM Kali) - rotating capture of SSH traffic from the attacker.
# Writes a new pcap every 2 seconds into the shared folder, so the attacker can convert it.
# Usage:  bash 1_capture_rotate.sh <IFACE> <ATTACKER_IP>
# Stop with Ctrl+C. The victim must have the shared folder mounted first (see ssh_plan.txt).

IFACE="$1"
ATTACKER_IP="$2"
OUT=/media/sf_F_DRIVE/all_10_csv_manual/live_detection/live/pcap

if [ -z "$IFACE" ] || [ -z "$ATTACKER_IP" ]; then
  echo "usage: bash 1_capture_rotate.sh <IFACE> <ATTACKER_IP>"; exit 1
fi
mkdir -p "$OUT" || { echo "cannot write $OUT - is the share mounted?"; exit 1; }

echo "capturing $IFACE host $ATTACKER_IP port 22 into $OUT every 2 s (Ctrl+C to stop)"
sudo tcpdump -i "$IFACE" -G 2 -w "$OUT/seg_%Y%m%d_%H%M%S.pcap" host "$ATTACKER_IP" and port 22
