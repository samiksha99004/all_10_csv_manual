"""Live network capture on Kali Linux -> /home/kali/samiksha_normal.csv

Same logic as capture.py (Windows version):
- creates samiksha_normal.csv with the exact 79-column header of the cleaned CICIDS2018 dataset
- sniffs the network, groups packets into flows, writes ONE row per finished flow
- fills only the columns that can be computed from the packets; the rest stay empty (Label too)
- each row is flushed when written, so the CSV stays readable after Ctrl+C

Run (needs root for packet capture):
  sudo python3 capture_kali.py                 (uses eth0; Ctrl+C to stop)
  sudo python3 capture_kali.py --iface eth0
Requirements:  sudo apt install -y python3-scapy   (or: pip install --break-system-packages scapy)
"""
import argparse
import csv
import math
import os

OUT_CSV = os.path.expanduser("~/samiksha_normal.csv")
IDLE_SECONDS = 30          # a flow with no packet for this long is finished
MAX_FLOW_SECONDS = 120     # a flow longer than this is finished too

# Exact header of cicids2018_full_cleaned.csv (79 columns, Timestamp already dropped).
HEADER = ("Dst Port,Protocol,Flow Duration,Tot Fwd Pkts,Tot Bwd Pkts,TotLen Fwd Pkts,TotLen Bwd Pkts,"
          "Fwd Pkt Len Max,Fwd Pkt Len Min,Fwd Pkt Len Mean,Fwd Pkt Len Std,Bwd Pkt Len Max,Bwd Pkt Len Min,"
          "Bwd Pkt Len Mean,Bwd Pkt Len Std,Flow Byts/s,Flow Pkts/s,Flow IAT Mean,Flow IAT Std,Flow IAT Max,"
          "Flow IAT Min,Fwd IAT Tot,Fwd IAT Mean,Fwd IAT Std,Fwd IAT Max,Fwd IAT Min,Bwd IAT Tot,Bwd IAT Mean,"
          "Bwd IAT Std,Bwd IAT Max,Bwd IAT Min,Fwd PSH Flags,Bwd PSH Flags,Fwd URG Flags,Bwd URG Flags,"
          "Fwd Header Len,Bwd Header Len,Fwd Pkts/s,Bwd Pkts/s,Pkt Len Min,Pkt Len Max,Pkt Len Mean,Pkt Len Std,"
          "Pkt Len Var,FIN Flag Cnt,SYN Flag Cnt,RST Flag Cnt,PSH Flag Cnt,ACK Flag Cnt,URG Flag Cnt,"
          "CWE Flag Count,ECE Flag Cnt,Down/Up Ratio,Pkt Size Avg,Fwd Seg Size Avg,Bwd Seg Size Avg,"
          "Fwd Byts/b Avg,Fwd Pkts/b Avg,Fwd Blk Rate Avg,Bwd Byts/b Avg,Bwd Pkts/b Avg,Bwd Blk Rate Avg,"
          "Subflow Fwd Pkts,Subflow Fwd Byts,Subflow Bwd Pkts,Subflow Bwd Byts,Init Fwd Win Byts,"
          "Init Bwd Win Byts,Fwd Act Data Pkts,Fwd Seg Size Min,Active Mean,Active Std,Active Max,Active Min,"
          "Idle Mean,Idle Std,Idle Max,Idle Min,Label").split(",")

FIN, SYN, RST, PSH, ACK, URG, ECE, CWR = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80


def stats(values):
    """(max, min, mean, std) of a list; zeros when empty."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return float(max(values)), float(min(values)), mean, math.sqrt(var)


class Flow:
    def __init__(self, t):
        self.first = t
        self.last = t
        self.packets = []   # (time, direction 0=fwd 1=bwd, ip_len, payload_len, hdr_len, flags, window)

    def add(self, t, direction, ip_len, payload, hdr, flags, window):
        self.packets.append((t, direction, ip_len, payload, hdr, flags, window))
        self.last = t


def features(f):
    """Columns that can be computed from the packets. Times in microseconds, like the training data."""
    pk = f.packets
    fwd = [p for p in pk if p[1] == 0]
    bwd = [p for p in pk if p[1] == 1]
    dur_us = (f.last - f.first) * 1e6
    dur_s = max(dur_us / 1e6, 1e-6)
    times = [p[0] for p in pk]
    iat = [(b - a) * 1e6 for a, b in zip(times, times[1:])]
    fiat = [(b[0] - a[0]) * 1e6 for a, b in zip(fwd, fwd[1:])]
    biat = [(b[0] - a[0]) * 1e6 for a, b in zip(bwd, bwd[1:])]
    fwd_len = [p[2] for p in fwd]
    bwd_len = [p[2] for p in bwd]
    all_len = [p[2] for p in pk]
    tot_bytes = sum(all_len)

    def flag_count(bit):
        return sum(1 for p in pk if p[5] & bit)

    fmax, fmin, fmean, fstd = stats(fwd_len)
    bmax, bmin, bmean, bstd = stats(bwd_len)
    amax, amin, amean, astd = stats(all_len)
    ivmax, ivmin, ivmean, ivstd = stats(iat)
    fimax, fimin, fimean, fistd = stats(fiat)
    bimax, bimin, bimean, bistd = stats(biat)
    fwd_bytes = sum(fwd_len)
    bwd_bytes = sum(bwd_len)
    first_fwd = fwd[0] if fwd else None
    first_bwd = bwd[0] if bwd else None
    return {
        "Dst Port": f.dport, "Protocol": f.proto, "Flow Duration": dur_us,
        "Tot Fwd Pkts": len(fwd), "Tot Bwd Pkts": len(bwd),
        "TotLen Fwd Pkts": fwd_bytes, "TotLen Bwd Pkts": bwd_bytes,
        "Fwd Pkt Len Max": fmax, "Fwd Pkt Len Min": fmin, "Fwd Pkt Len Mean": fmean, "Fwd Pkt Len Std": fstd,
        "Bwd Pkt Len Max": bmax, "Bwd Pkt Len Min": bmin, "Bwd Pkt Len Mean": bmean, "Bwd Pkt Len Std": bstd,
        "Flow Byts/s": tot_bytes / dur_s, "Flow Pkts/s": len(pk) / dur_s,
        "Flow IAT Mean": ivmean, "Flow IAT Std": ivstd, "Flow IAT Max": ivmax, "Flow IAT Min": ivmin,
        "Fwd IAT Tot": sum(fiat), "Fwd IAT Mean": fimean, "Fwd IAT Std": fistd, "Fwd IAT Max": fimax, "Fwd IAT Min": fimin,
        "Bwd IAT Tot": sum(biat), "Bwd IAT Mean": bimean, "Bwd IAT Std": bistd, "Bwd IAT Max": bimax, "Bwd IAT Min": bimin,
        "Fwd PSH Flags": sum(1 for p in fwd if p[5] & PSH), "Bwd PSH Flags": sum(1 for p in bwd if p[5] & PSH),
        "Fwd URG Flags": sum(1 for p in fwd if p[5] & URG), "Bwd URG Flags": sum(1 for p in bwd if p[5] & URG),
        "Fwd Header Len": sum(p[4] for p in fwd), "Bwd Header Len": sum(p[4] for p in bwd),
        "Fwd Pkts/s": len(fwd) / dur_s, "Bwd Pkts/s": len(bwd) / dur_s,
        "Pkt Len Min": amin, "Pkt Len Max": amax, "Pkt Len Mean": amean, "Pkt Len Std": astd, "Pkt Len Var": astd ** 2,
        "FIN Flag Cnt": flag_count(FIN), "SYN Flag Cnt": flag_count(SYN), "RST Flag Cnt": flag_count(RST),
        "PSH Flag Cnt": flag_count(PSH), "ACK Flag Cnt": flag_count(ACK), "URG Flag Cnt": flag_count(URG),
        "CWE Flag Count": flag_count(CWR), "ECE Flag Cnt": flag_count(ECE),
        "Down/Up Ratio": (bwd_bytes / fwd_bytes) if fwd_bytes else 0.0,
        "Pkt Size Avg": tot_bytes / len(pk) if pk else 0.0,
        "Fwd Seg Size Avg": fmean, "Bwd Seg Size Avg": bmean,
        "Subflow Fwd Pkts": len(fwd), "Subflow Fwd Byts": fwd_bytes,
        "Subflow Bwd Pkts": len(bwd), "Subflow Bwd Byts": bwd_bytes,
        "Init Fwd Win Byts": first_fwd[6] if first_fwd else 0, "Init Bwd Win Byts": first_bwd[6] if first_bwd else 0,
        "Fwd Act Data Pkts": sum(1 for p in fwd if p[3] > 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="eth0", help="network interface (default eth0)")
    args = ap.parse_args()

    from scapy.all import IP, TCP, UDP, sniff

    header = HEADER
    new_file = not os.path.exists(OUT_CSV)
    out = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=header, restval="", extrasaction="ignore")
    if new_file:
        writer.writeheader()
        out.flush()
    print(f"writing to {OUT_CSV} ({len(header)} columns) on {args.iface}. Ctrl+C to stop.", flush=True)

    flows = {}
    written = 0

    def finish(key):
        nonlocal written
        f = flows.pop(key)
        writer.writerow(features(f))
        out.flush()
        written += 1

    def handle(pkt):
        if IP not in pkt:
            return
        ip = pkt[IP]
        if TCP in pkt:
            proto, sp, dp = 6, pkt[TCP].sport, pkt[TCP].dport
            flags, window = int(pkt[TCP].flags), pkt[TCP].window
            hdr = len(pkt[TCP]) - len(pkt[TCP].payload)
            payload = len(bytes(pkt[TCP].payload))
        elif UDP in pkt:
            proto, sp, dp = 17, pkt[UDP].sport, pkt[UDP].dport
            flags, window, hdr = 0, 0, 8
            payload = len(bytes(pkt[UDP].payload))
        else:
            return
        t = float(pkt.time)
        key = (ip.src, ip.dst, sp, dp, proto)
        rkey = (ip.dst, ip.src, dp, sp, proto)
        if key in flows:
            f, direction = flows[key], 0
        elif rkey in flows:
            f, direction, key = flows[rkey], 1, rkey
        else:
            f = Flow(t)
            f.dport, f.proto = dp, proto
            flows[key] = f
            direction = 0
        f.add(t, direction, int(ip.len), payload, hdr, flags, window)
        for k in [k for k, v in flows.items() if (t - v.last > IDLE_SECONDS) or (t - v.first > MAX_FLOW_SECONDS)]:
            finish(k)

    try:
        sniff(iface=args.iface, prn=handle, store=False)
    except KeyboardInterrupt:
        pass
    finally:
        for k in list(flows):
            finish(k)
        out.close()
        print(f"\nstopped. rows written this run: {written}. file: {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
