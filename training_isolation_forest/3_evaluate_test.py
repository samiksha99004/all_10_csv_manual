"""Step 3 - Test the Isolation Forest on the 20% test split (never used before).

Scores the test rows in chunks, applies the threshold chosen in step 2 and
reports, for every class, the share handled correctly: attacks caught (recall)
and Benign not flagged. Isolation Forest only says "normal" or "attack"; it
cannot name the attack type, so per-attack numbers use the true labels.

Also shows the trade-off: per-attack recall at fixed false-alarm rates, with
those thresholds taken from the TRAINING Benign scores (no tuning on test).

Outputs: test_report.txt, test_metrics.json, per_attack_recall.png, score_distribution.png
"""
from if_common import CONFIG_PATH, HERE, MODEL_PATH, TARGET, anomaly_scores, load, meta  # keep first (WMI)

import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FALSE_ALARM_LEVELS = [0.01, 0.02, 0.05, 0.10, 0.20]


def main():
    t0 = time.time()
    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
    classes, benign, thr = cfg["classes"], cfg["benign_code"], cfg["threshold"]
    keep = np.array(cfg["keep_columns"])
    model = joblib.load(MODEL_PATH)
    X, y = load("X_test"), load("y_test", mmap=False)

    print(f"Scoring {len(X):,} test rows in chunks ...", flush=True)
    t1 = time.time()
    scores = anomaly_scores(model, X, keep)
    secs = time.time() - t1

    is_attack = y != benign
    flagged = scores >= thr
    tp = int((flagged & is_attack).sum()); fn = int((~flagged & is_attack).sum())
    fp = int((flagged & ~is_attack).sum()); tn = int((~flagged & ~is_attack).sum())
    acc = (tp + tn) / len(y)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    per = {}
    for c in range(len(classes)):
        m = y == c
        r = float(1 - flagged[m].mean()) if c == benign else float(flagged[m].mean())
        per[classes[c]] = {"rows": int(m.sum()), "rate": round(r, 6), "meets_95": r >= TARGET}
    n_ok = sum(v["meets_95"] for v in per.values())

    # trade-off table: thresholds fixed from training Benign scores
    tr_scores, tr_y = load("train_scores", mmap=False), load("y_train", mmap=False)
    tr_benign = tr_scores[tr_y == benign]
    sweep = {}
    for far in FALSE_ALARM_LEVELS:
        t = float(np.quantile(tr_benign, 1 - far))
        f = scores >= t
        sweep[f"{int(far * 100)}%"] = {classes[c]: round(float(f[y == c].mean()), 4)
                                       for c in range(len(classes)) if c != benign}
        sweep[f"{int(far * 100)}%"]["_test_false_alarm"] = round(float(f[~is_attack].mean()), 4)

    json.dump({"threshold": thr, "test_rows": int(len(y)), "accuracy": acc, "attack_precision": prec,
               "attack_recall": rec, "attack_f1": f1, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
               "classes_meeting_95": n_ok, "per_class": per, "recall_at_false_alarm_rate": sweep,
               "flows_per_second": int(len(y) / secs)},
              open(os.path.join(HERE, "test_metrics.json"), "w"), indent=2)

    order = sorted(per, key=lambda k: -per[k]["rows"])
    lines = [f"Isolation Forest - test split ({len(y):,} rows, never used for training or the threshold)",
             "=" * 78,
             f"Overall accuracy (normal vs attack): {acc:.4f}",
             f"Attack detection: precision {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}",
             f"Confusion: attacks caught {tp:,} | missed {fn:,} | false alarms {fp:,} | normal passed {tn:,}",
             f"Classes at >= 95%: {n_ok}/{len(classes)}    speed: {int(len(y) / secs):,} flows/s", "",
             f"{'class':<26}{'test rows':>10}{'rate':>9}   (attacks: caught; Benign: not flagged)"]
    for k in order:
        v = per[k]
        lines.append(f"{k:<26}{v['rows']:>10,}{v['rate'] * 100:>8.1f}%{'' if v['meets_95'] else '   < 95%'}")
    lines += ["", "Attack recall at fixed false-alarm rates (thresholds from training Benign scores):", ""]
    atk = [k for k in order if k != "Benign"]
    lines.append(f"{'class':<26}" + "".join(f"{lv:>8}" for lv in sweep))
    for k in atk:
        lines.append(f"{k:<26}" + "".join(f"{sweep[lv][k] * 100:>7.1f}%" for lv in sweep))
    lines.append(f"{'(test false alarms)':<26}" + "".join(f"{sweep[lv]['_test_false_alarm'] * 100:>7.1f}%" for lv in sweep))
    report = "\n".join(lines)
    print("\n" + report)
    open(os.path.join(HERE, "test_report.txt"), "w", encoding="utf-8").write(report + "\n")

    # charts
    rates = [per[k]["rate"] * 100 for k in order[::-1]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(order[::-1], rates, color=["#1f9d55" if r >= 95 else "#c1443a" for r in rates])
    ax.axvline(95, color="#333", ls="--", lw=1); ax.set_xlim(0, 100); ax.set_xlabel("% handled correctly")
    ax.set_title("Isolation Forest per-class result on the test set (green = meets 95%)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "per_attack_recall.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(scores.min(), scores.max(), 120)
    ax.hist(scores[~is_attack], bins=bins, alpha=0.6, label="Benign", density=True)
    ax.hist(scores[is_attack], bins=bins, alpha=0.6, label="Attacks", density=True)
    ax.axvline(thr, color="k", ls="--", label="threshold"); ax.set_xlabel("anomaly score (higher = more unusual)")
    ax.legend(); ax.set_title("Anomaly scores on the test set")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "score_distribution.png"), dpi=110); plt.close(fig)
    print(f"\nSaved report, metrics and charts  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
