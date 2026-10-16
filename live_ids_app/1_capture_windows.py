"""Step 1 (Windows) - rotating packet capture using scapy (no tcpdump on this machine).
Writes a new pcap every WINDOW_SECONDS (5) seconds into pcap/, so step 2 can convert each one.

MUST be run from an Administrator Command Prompt / PowerShell - packet capture needs admin
rights on Windows (Npcap). Needs Npcap installed (already confirmed working on this machine).

Usage:
  python 1_capture_windows.py                 (uses scapy's default interface)
  python 1_capture_windows.py --iface "Wi-Fi"  (pick a specific interface; see: ipconfig)
Stop with Ctrl+C.

No filter is applied - this captures ALL traffic on the interface, so the app can recognise
any of the 15 trained classes, not just one attack type. Pass --filter to narrow it
(scapy/BPF syntax, e.g. --filter "host 192.168.1.50").
"""
from app_common import PCAP_DIR, WINDOW_SECONDS  # keep first (WMI patch)

import sys
sys.stdout.reconfigure(line_buffering=True, write_through=True)  # force real-time printing
sys.stderr.reconfigure(line_buffering=True, write_through=True)

import argparse
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default=None, help="interface name (default: scapy's default)")
    ap.add_argument("--filter", default=None, help="BPF filter, e.g. 'host 192.168.1.50'")
    args = ap.parse_args()

    from scapy.all import sniff, wrpcap

    iface = args.iface
    if not iface:
        # Scapy's own default-interface pick is unreliable on Windows (it can land on a
        # VPN/virtual/loopback adapter with no real traffic). Prefer a physical adapter that
        # actually has a routable IPv4 address instead.
        from scapy.arch.windows import get_windows_if_list
        bad_name = ("loopback", "virtual", "vpn", "bluetooth", "hyper-v", "npcap loopback")
        candidates = [i for i in get_windows_if_list()
                     if i.get("ips") and not any(b in i.get("name", "").lower() for b in bad_name)
                     and not any(b in i.get("description", "").lower() for b in bad_name)
                     and any(not ip.startswith(("127.", "169.254.")) and ":" not in ip for ip in i["ips"])]
        if candidates:
            iface = candidates[0]["name"]

    os.makedirs(PCAP_DIR, exist_ok=True)
    print(f"writing a new pcap every {WINDOW_SECONDS}s into {PCAP_DIR}", flush=True)
    print("NOTE: this must run as Administrator, or sniff() will fail to open the interface.", flush=True)
    print(f"interface: {iface or '(scapy default - pass --iface if this captures nothing)'}", flush=True)
    print("capturing now - Ctrl+C to stop.\n", flush=True)

    try:
        while True:
            kwargs = {"timeout": WINDOW_SECONDS, "store": True}
            if iface:
                kwargs["iface"] = iface
            if args.filter:
                kwargs["filter"] = args.filter
            packets = sniff(**kwargs)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(PCAP_DIR, f"seg_{stamp}.pcap")
            if len(packets) > 0:
                wrpcap(path, packets)
                print(f"[{time.strftime('%H:%M:%S')}] wrote {path} ({len(packets)} packets)", flush=True)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] no packets this window", flush=True)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
