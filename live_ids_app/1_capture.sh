#!/bin/bash
# Step 1 (run on the machine doing live detection, e.g. the Kali "victim") - rotating capture.
# Writes a new pcap every WINDOW_SECONDS (5) seconds, so step 2 can convert each one.
#
# Usage:  sudo bash 1_capture.sh <IFACE>
#   <IFACE>  network interface to watch, e.g. eth0 (check with: ip -br a)
# Stop with Ctrl+C.
#
# No port/host filter is applied - this captures ALL traffic on the interface, because this
# app detects any of the 15 trained classes (DoS, DDoS, Bot, brute force, etc.), not SSH only.
# Edit the BPF filter below if you want to narrow it (e.g. to one host).

WINDOW_SECONDS=5
IFACE="$1"
OUT="$HOME/live_ids_app/pcap"
BPF=""          # e.g. BPF="host 192.168.1.50" to capture only one host's traffic

if [ -z "$IFACE" ]; then
  echo "usage: sudo bash 1_capture.sh <IFACE>"; exit 1
fi
mkdir -p "$OUT"

echo "capturing on $IFACE into $OUT, one file every ${WINDOW_SECONDS}s (Ctrl+C to stop)"
# -Z root: writes the rotated files as root (fixes "Permission denied" seen when tcpdump drops
# privileges to a low-privilege user mid-run - see the project's earlier debugging of this).
tcpdump -Z root -i "$IFACE" -G "$WINDOW_SECONDS" -w "$OUT/seg_%Y%m%d_%H%M%S.pcap" $BPF
