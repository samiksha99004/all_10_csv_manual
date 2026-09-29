"""Step 4 - Test the two-level pipeline on the 20% test split (unseen by both).

For each test flow:
  level 1 -> not attack? final label = Benign.
           -> attack?     level 2 names which of the 14 attacks.
Reports overall accuracy, per-class precision/recall/F1, and a 15x15 confusion
matrix.

Outputs: final_report.txt, final_metrics.json, confusion_matrix.csv/.png
"""
from xgb_common import CHUNK_ROWS, DATA, HERE, L1_PATH, L2_PATH, load, meta  # keep first (WMI)

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def main():
    t0 = time.time()
    m = meta(); classes, benign = m["classes"], m["benign_code"]
    attack_classes = json.load(open(os.path.join(DATA, "attack_classes.json"), encoding="utf-8"))
    attack_to_code = {name: classes.index(name) for name in attack_classes.values()}
    a2c = np.array([attack_to_code[attack_classes[str(i)]] for i in range(len(attack_classes))])  # level2 idx -> global code
    k = len(classes)

    l1 = xgb.XGBClassifier(); l1.load_model(L1_PATH)
    l2 = xgb.XGBClassifier(); l2.load_model(L2_PATH)
    X, y = load("X_test"), load("y_test", mmap=False)

    print(f"Scoring {len(X):,} test rows in chunks ...", flush=True)
    pred = np.empty(len(X), dtype=np.int64)
    for s in range(0, len(X), CHUNK_ROWS):
        xb = np.ascontiguousarray(X[s:s + CHUNK_ROWS])
        is_attack = l1.predict(xb).astype(bool)
        out = np.full(len(xb), benign, dtype=np.int64)
        if is_attack.any():
            out[is_attack] = a2c[l2.predict(xb[is_attack])]
        pred[s:s + CHUNK_ROWS] = out

    acc = float((pred == y).mean())
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(k), zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(k))
    per = {classes[c]: {"test_rows": int(s[c]), "precision": float(p[c]), "recall": float(r[c]),
                        "f1": float(f[c])} for c in range(k)}
    n_ok = sum(v["recall"] >= 0.95 for v in per.values())

    json.dump({"overall_accuracy": acc, "classes_meeting_95": n_ok, "per_class": per},
              open(os.path.join(HERE, "final_metrics.json"), "w"), indent=2)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(HERE, "confusion_matrix.csv"))

    order = sorted(range(k), key=lambda c: -s[c])
    L = [f"Two-level XGBoost - test split ({len(y):,} rows)", "=" * 70,
         f"Overall accuracy: {acc:.4f}    classes at >= 95% recall: {n_ok}/{k}", "",
         f"{'class':<26}{'test rows':>10}{'precision':>11}{'recall':>9}{'F1':>8}"]
    for c in order:
        v = per[classes[c]]
        L.append(f"{classes[c]:<26}{v['test_rows']:>10,}{v['precision']*100:>10.1f}%{v['recall']*100:>8.1f}%"
                 f"{v['f1']:>8.3f}{'' if v['recall'] >= 0.95 else '   < 95%'}")
    report = "\n".join(L)
    print("\n" + report)
    open(os.path.join(HERE, "final_report.txt"), "w", encoding="utf-8").write(report + "\n")

    share = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(12, 10)); ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k), classes, rotation=90, fontsize=7); ax.set_yticks(range(k), classes, fontsize=7)
    for i in range(k):
        for j in range(k):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=5.5,
                        color="white" if share[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Two-level XGBoost confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "confusion_matrix.png"), dpi=120); plt.close(fig)
    print(f"\nSaved report, metrics and confusion matrix  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
