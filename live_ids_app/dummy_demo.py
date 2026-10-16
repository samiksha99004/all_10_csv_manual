"""DUMMY demo printer - NOT a real detector. Reads no network data, loads no model - prints
fake NETWORK STATUS lines every 3 seconds in the same style as 3_predict.py, so you can show
teammates what live output looks like without needing a live capture running at all.

First 30 seconds: Benign 100%. After that: a changing mix of DoS, DDoS and Normal percentages.

Usage: python dummy_demo.py
Stop with Ctrl+C.
"""
import random
import sys
import time

sys.stdout.reconfigure(line_buffering=True, write_through=True)

INTERVAL_SECONDS = 3
BENIGN_PHASE_SECONDS = 30
TOTAL_FLOWS = 12          # fake flow count per window, just for display


def random_attack_breakdown():
    """DoS + DDoS + Normal percentages, summing to 100 - varies a bit each tick for a 'live' feel."""
    dos = random.randint(35, 60)
    ddos = random.randint(10, 95 - dos)
    normal = 100 - dos - ddos
    return {"DoS": dos, "DDoS": ddos, "Normal": normal}


def print_window(stamp, breakdown, cumulative):
    verdict = max(breakdown, key=breakdown.get)
    print(f"[{stamp}] window: {TOTAL_FLOWS} flows captured, {TOTAL_FLOWS} scored, 0 dropped", flush=True)
    print(f"  NETWORK STATUS: {verdict}", flush=True)
    for name, pct in breakdown.items():
        count = round(TOTAL_FLOWS * pct / 100)
        cumulative[name] = cumulative.get(name, 0) + count
        print(f"    {name:<28}{count:>5}  {pct:5.1f}%", flush=True)
    total = sum(cumulative.values())
    print(f"  cumulative: {total} flows scored, 0 dropped, since start", flush=True)
    print("-" * 70, flush=True)


def main():
    print("DUMMY DEMO - fake values only, no real network/model involved.", flush=True)
    print(f"printing every {INTERVAL_SECONDS}s. Ctrl+C to stop.\n", flush=True)
    start = time.time()
    cumulative = {}
    try:
        while True:
            stamp = time.strftime("%H:%M:%S")
            elapsed = time.time() - start
            breakdown = {"Benign": 100} if elapsed < BENIGN_PHASE_SECONDS else random_attack_breakdown()
            print_window(stamp, breakdown, cumulative)
            time.sleep(INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nstopped.")
        print("final cumulative counts:")
        for name, c in sorted(cumulative.items(), key=lambda x: -x[1]):
            print(f"  {name:<28}{c:>6}")


if __name__ == "__main__":
    main()
