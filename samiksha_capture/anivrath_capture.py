"""Live capture -> anivrath.csv : the cleaned CICIDS2018 layout (79 columns).

Captures every packet on the network interface, groups them into flows (5-tuple), and writes ONE
row per finished flow with all 78 features computed from the packets. Label is left EMPTY on
purpose: the program does not know if the traffic is normal or attack. The rows are never
labelled by this program.

Rows are written and flushed as each flow ends, and when the program stops (Ctrl+C) the open
flows are written too. Windows: run as Administrator with Npcap installed.
Kali: run with sudo.

  Windows:  python anivrath_capture.py
  Kali:     sudo python3 anivrath_capture.py --iface eth0

Things that must be checked against the converter source (not verifiable from this PC):
  - ACTIVITY_TIMEOUT_US (used for Active and Idle columns)
  - Fwd Seg Size Min (defined here as the smallest forward TCP header length)
"""
import argparse
import csv
import math
import os
import sys

if sys.platform.startswith("win"):
    import platform
    platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())  # Windows WMI hangs imports

ACTIVITY_TIMEOUT_US = 5_000_000   # gap that ends an active period (CHECK against converter source)
IDLE_SECONDS = 30                 # a flow with no packet this long is finished
MAX_FLOW_SECONDS = 120            # a flow longer than this is finished too

if sys.platform.startswith("win"):
    OUT_CSV = r"F:\anivrath.csv"
else:
    _home = os.path.expanduser("~")
    if os.geteuid() == 0 and os.environ.get("SUDO_USER"):      # run with sudo: use the user's home
        _home = os.path.join("/home", os.environ["SUDO_USER"])
    OUT_CSV = os.path.join(_home, "anivrath.csv")

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
    """(max, min, mean, std); zeros when empty."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return float(max(values)), float(min(values)), mean, math.sqrt(var)


class Flow:
    def __init__(self, t, dport, proto):
        self.first = t
        self.last = t
        self.dport = dport
        self.proto = proto
        self.pk = []        # (time, dir 0=fwd 1=bwd, payload, hdr_len, flags, window)
        self.active = []    # finished active periods (us)
        self.idle = []      # finished idle gaps (us)
        self.act_start = t
        self.act_end = t

    def add(self, t, d, payload, hdr, flags, window):
        # active / idle bookkeeping (gap-based, see ACTIVITY_TIMEOUT_US)
        gap = (t - self.act_end) * 1e6
        if gap > ACTIVITY_TIMEOUT_US and len(self.pk) > 0:
            if self.act_end > self.act_start:
                self.active.append((self.act_end - self.act_start) * 1e6)
            self.idle.append(gap)
            self.act_start = t
        self.act_end = t
        self.pk.append((t, d, payload, hdr, flags, window))
        self.last = t

    def finish_active(self):
        if self.act_end > self.act_start:
            self.active.append((self.act_end - self.act_start) * 1e6)
            self.act_start = self.act_end


def features(f):
    f.finish_active()
    pk = f.pk
    fwd = [p for p in pk if p[1] == 0]
    bwd = [p for p in pk if p[1] == 1]
    dur_us = (f.last - f.first) * 1e6
    dur_s = max(dur_us / 1e6, 1e-6)
    times = [p[0] for p in pk]
    iat = [(b - a) * 1e6 for a, b in zip(times, times[1:])]
    fiat = [(b[0] - a[0]) * 1e6 for a, b in zip(fwd, fwd[1:])]
    biat = [(b[0] - a[0]) * 1e6 for a, b in zip(bwd, bwd[1:])]
    fl = [p[2] for p in fwd]
    bl = [p[2] for p in bwd]
    al = [p[2] for p in pk]
    fwd_bytes, bwd_bytes = sum(fl), sum(bl)
    tot_bytes = fwd_bytes + bwd_bytes
    fmax, fmin, fmean, fstd = stats(fl)
    bmax, bmin, bmean, bstd = stats(bl)
    amax, amin, amean, astd = stats(al)
    ivmax, ivmin, ivmean, ivstd = stats(iat)
    fimax, fimin, fimean, fistd = stats(fiat)
    bimax, bimin, bimean, bistd = stats(biat)
    amax_act, amin_act, amean_act, astd_act = stats(f.active)
    imax, imin, imean, istd = stats(f.idle)
    fwd_hdr = [p[3] for p in fwd]
    first_fwd = fwd[0] if fwd else None
    first_bwd = bwd[0] if bwd else None

    def flag(bit):
        return sum(1 for p in pk if p[4] & bit)

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
        "Fwd PSH Flags": sum(1 for p in fwd if p[4] & PSH), "Bwd PSH Flags": sum(1 for p in bwd if p[4] & PSH),
        "Fwd URG Flags": sum(1 for p in fwd if p[4] & URG), "Bwd URG Flags": sum(1 for p in bwd if p[4] & URG),
        "Fwd Header Len": sum(fwd_hdr), "Bwd Header Len": sum(p[3] for p in bwd),
        "Fwd Pkts/s": len(fwd) / dur_s, "Bwd Pkts/s": len(bwd) / dur_s,
        "Pkt Len Min": amin, "Pkt Len Max": amax, "Pkt Len Mean": amean, "Pkt Len Std": astd, "Pkt Len Var": astd ** 2,
        "FIN Flag Cnt": flag(FIN), "SYN Flag Cnt": flag(SYN), "RST Flag Cnt": flag(RST),
        "PSH Flag Cnt": flag(PSH), "ACK Flag Cnt": flag(ACK), "URG Flag Cnt": flag(URG),
        "CWE Flag Count": flag(CWR), "ECE Flag Cnt": flag(ECE),
        "Down/Up Ratio": (bwd_bytes / fwd_bytes) if fwd_bytes else 0.0,
        "Pkt Size Avg": tot_bytes / len(pk) if pk else 0.0,
        "Fwd Seg Size Avg": fmean, "Bwd Seg Size Avg": bmean,
        # bulk features: always 0 in the cleaned training data, so they are written as 0
        "Fwd Byts/b Avg": 0, "Fwd Pkts/b Avg": 0, "Fwd Blk Rate Avg": 0,
        "Bwd Byts/b Avg": 0, "Bwd Pkts/b Avg": 0, "Bwd Blk Rate Avg": 0,
        "Subflow Fwd Pkts": len(fwd), "Subflow Fwd Byts": fwd_bytes,
        "Subflow Bwd Pkts": len(bwd), "Subflow Bwd Byts": bwd_bytes,
        "Init Fwd Win Byts": first_fwd[5] if first_fwd else -1,
        "Init Bwd Win Byts": first_bwd[5] if first_bwd else -1,
        "Fwd Act Data Pkts": sum(1 for p in fwd if p[2] > 0),
        "Fwd Seg Size Min": min(fwd_hdr) if fwd_hdr else 0,
        "Active Mean": amean_act, "Active Std": astd_act, "Active Max": amax_act, "Active Min": amin_act,
        "Idle Mean": imean, "Idle Std": istd, "Idle Max": imax, "Idle Min": imin,
        "Label": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default=None, help="network interface (Kali: eth0)")
    args = ap.parse_args()
    if args.iface is None and not sys.platform.startswith("win"):
        args.iface = "eth0"

    from scapy.all import IP, TCP, UDP, sniff

    new_file = not os.path.exists(OUT_CSV)
    out = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=HEADER, restval="", extrasaction="ignore")
    if new_file:
        writer.writeheader()
        out.flush()
    print(f"writing to {OUT_CSV} ({len(HEADER)} columns). Ctrl+C to stop.", flush=True)

    flows, written = {}, 0

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
            flags, window = int(pkt[TCP].flags), int(pkt[TCP].window)
            hdr = int(pkt[TCP].dataofs or 5) * 4
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
            f, d = flows[key], 0
        elif rkey in flows:
            f, d, key = flows[rkey], 1, rkey
        else:
            f, d = Flow(t, dp, proto), 0
            flows[key] = f
        f.add(t, d, payload, hdr, flags, window)
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
