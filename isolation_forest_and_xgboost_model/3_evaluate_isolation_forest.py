"""Step 3 - Test the Isolation Forest on the 20% test split (never used before).

Uses the threshold from step 2, which is fixed so at most 5% of normal traffic
is flagged. Reports normal accuracy and attack recall together, per class, so a
high attack number can never be bought by flagging normal traffic.

Also shows the trade-off: normal accuracy and attack recall at several
false-alarm ceilings (thresholds taken from the TRAINING Benign scores, so the
test set is not used to choose them).

Outputs: test_report.txt, test_metrics.json, per_attack_recall.png, score_distribution.png
"""
from if_common import CONFIG_PATH, HERE, MODEL_PATH, SCALER_PATH, TARGET, anomaly_scores, load  # keep first (WMI)

import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

FALSE_ALARM_CEILINGS = [0.01, 0.02, 0.05, 0.10]


def main():
    t0 = time.time()
    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
    classes, benign, thr = cfg["classes"], cfg["benign_code"], cfg["threshold"]
    keep = np.array(cfg["keep_columns"])
    model, scaler = joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    X, y = load("X_test"), load("y_test", mmap=False)

    print(f"Scoring {len(X):,} test rows in chunks ...", flush=True)
    t1 = time.time()
    scores = anomaly_scores(model, X, keep, scaler)
    secs = time.time() - t1

    is_attack = y != benign
    flagged = scores >= thr
    benign_acc = float(1 - flagged[~is_attack].mean())
    attack_rec = float(flagged[is_attack].mean())
    acc = float((flagged == is_attack).mean())
    bal_acc = (benign_acc + attack_rec) / 2
    auc = float(roc_auc_score(is_attack, scores))

    per, n_ok = {}, 0
    for c in range(len(classes)):
        m = y == c
        r = float(1 - flagged[m].mean()) if c == benign else float(flagged[m].mean())
        per[classes[c]] = {"rows": int(m.sum()), "rate": round(r, 6), "meets_95": r >= TARGET}
        n_ok += r >= TARGET

    tr_scores, tr_y = load("train_scores", mmap=False), load("y_train", mmap=False)
    tr_benign = tr_scores[tr_y == benign]
    sweep = {}
    for far in FALSE_ALARM_CEILINGS:
        t = float(np.quantile(tr_benign, 1 - far))
        f = scores >= t
        sweep[f"{int(far * 100)}%"] = {"normal_accuracy": round(float(1 - f[~is_attack].mean()), 4),
                                       "attack_recall": round(float(f[is_attack].mean()), 4)}

    json.dump({"threshold": thr, "test_rows": int(len(y)), "normal_accuracy": benign_acc,
               "attack_recall": attack_rec, "balanced_accuracy": bal_acc, "roc_auc": auc,
               "overall_accuracy": acc, "classes_meeting_95": n_ok,
               "per_class": per, "tradeoff_by_false_alarm_ceiling": sweep,
               "flows_per_second": int(len(y) / secs)},
              open(os.path.join(HERE, "test_metrics.json"), "w"), indent=2)

    order = sorted(per, key=lambda k: -per[k]["rows"])
    L = [f"Isolation Forest - test split ({len(y):,} rows, never used for training or the threshold)",
         "=" * 78,
         f"Normal accuracy: {benign_acc * 100:.1f}%    Attack recall: {attack_rec * 100:.1f}%    "
         f"Balanced accuracy: {bal_acc * 100:.1f}%",
         f"Overall accuracy: {acc * 100:.1f}%    ROC-AUC: {auc:.3f}    classes at >= 95%: {n_ok}/{len(classes)}",
         f"Speed: {int(len(y) / secs):,} flows/s", "",
         f"{'class':<26}{'test rows':>10}{'rate':>9}   (attacks: caught; Benign: not flagged)"]
    for k in order:
        v = per[k]
        L.append(f"{k:<26}{v['rows']:>10,}{v['rate'] * 100:>8.1f}%{'' if v['meets_95'] else '   < 95%'}")
    L += ["", "Trade-off - normal accuracy vs attack recall at each false-alarm ceiling:", "",
          f"{'false-alarm ceiling':<22}{'normal acc':>12}{'attack recall':>15}"]
    for lv, d in sweep.items():
        L.append(f"{lv:<22}{d['normal_accuracy'] * 100:>11.1f}%{d['attack_recall'] * 100:>14.1f}%")
    report = "\n".join(L)
    print("\n" + report)
    open(os.path.join(HERE, "test_report.txt"), "w", encoding="utf-8").write(report + "\n")

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
    ax.axvline(thr, color="k", ls="--", label="threshold"); ax.legend()
    ax.set_xlabel("anomaly score (higher = more unusual)"); ax.set_title("Anomaly scores on the test set")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "score_distribution.png"), dpi=110); plt.close(fig)
    print(f"\nSaved report, metrics and charts  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
