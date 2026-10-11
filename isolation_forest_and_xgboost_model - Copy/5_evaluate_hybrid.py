"""Step 5 - Test the hybrid on the 20% test split (never used for training).

For every test flow:
  1. Isolation Forest scores it. Below the hybrid threshold -> "Benign", done.
  2. Otherwise the flow is suspicious and goes to XGBoost, which names the attack
     (or answers "Benign", cancelling a false alarm).

Reports stage 1 on its own (normal vs attack), the final hybrid result per class
(how many of 100 attacks were caught AND named correctly), overall accuracy,
precision / recall / F1, and a 15x15 confusion matrix. XGBoost on its own is
shown for comparison.

Outputs: hybrid_report.txt, hybrid_metrics.json, hybrid_confusion_matrix.csv/.png
"""
from if_common import (CHUNK_ROWS, CONFIG_PATH, HERE, MODEL_PATH, SCALER_PATH, TARGET, XGB_PATH,
                       anomaly_scores, load)  # keep first (WMI)

import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def main():
    t0 = time.time()
    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
    classes, benign = cfg["classes"], cfg["benign_code"]
    keep, thr = np.array(cfg["keep_columns"]), cfg["threshold"]
    k = len(classes)
    forest, scaler = joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    xgbm = xgb.XGBClassifier(); xgbm.load_model(XGB_PATH)
    X, y = load("X_test"), load("y_test", mmap=False)

    print(f"Stage 1: Isolation Forest scoring {len(X):,} test rows ...", flush=True)
    scores = anomaly_scores(forest, X, keep, scaler)
    suspicious = scores >= thr

    print("Stage 2: XGBoost naming the attacks ...", flush=True)
    xgb_pred = np.empty(len(X), dtype=np.int64)
    for s in range(0, len(X), CHUNK_ROWS):
        xgb_pred[s:s + CHUNK_ROWS] = xgbm.predict(np.ascontiguousarray(X[s:s + CHUNK_ROWS][:, keep]))
    hybrid = np.where(suspicious, xgb_pred, benign)

    is_attack = y != benign
    s1 = {"attacks_passed_to_stage2": int((suspicious & is_attack).sum()),
          "attacks_missed_by_stage1": int((~suspicious & is_attack).sum()),
          "benign_false_alarms_stage1": int((suspicious & ~is_attack).sum()),
          "false_alarms_cancelled_by_xgboost": int((suspicious & ~is_attack & (xgb_pred == benign)).sum()),
          "stage1_accuracy": float((suspicious == is_attack).mean()),
          "stage1_attack_recall": float(suspicious[is_attack].mean())}

    def summarise(pred):
        p, r, f, n = precision_recall_fscore_support(y, pred, labels=range(k), zero_division=0)
        return {"accuracy": float((pred == y).mean()),
                "per_class": {classes[c]: {"test_rows": int(n[c]), "precision": float(p[c]), "recall": float(r[c]),
                                           "f1": float(f[c])} for c in range(k)}}
    res_h, res_x = summarise(hybrid), summarise(xgb_pred)
    n_ok = sum(v["recall"] >= TARGET for v in res_h["per_class"].values())
    cm = confusion_matrix(y, hybrid, labels=range(k))

    json.dump({"threshold": thr, "stage1": s1, "hybrid": res_h, "xgboost_alone": res_x,
               "classes_meeting_95": n_ok}, open(os.path.join(HERE, "hybrid_metrics.json"), "w"), indent=2)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(HERE, "hybrid_confusion_matrix.csv"))

    order = sorted(range(k), key=lambda c: -res_h["per_class"][classes[c]]["test_rows"])
    L = [f"Hybrid Isolation Forest -> XGBoost, test split ({len(y):,} rows never used in training)", "=" * 84,
         f"HYBRID overall accuracy: {res_h['accuracy']:.4f}   (XGBoost alone: {res_x['accuracy']:.4f})",
         f"Classes with recall >= 95%: {n_ok}/{k}", "",
         "Stage 1 - Isolation Forest (normal vs attack):",
         f"  accuracy {s1['stage1_accuracy']:.4f}, attacks sent on {s1['stage1_attack_recall']:.2%}",
         f"  attacks missed at stage 1: {s1['attacks_missed_by_stage1']:,}",
         f"  false alarms: {s1['benign_false_alarms_stage1']:,}, of which XGBoost cancelled "
         f"{s1['false_alarms_cancelled_by_xgboost']:,}", "",
         f"{'class':<26}{'test rows':>10}{'precision':>11}{'recall':>9}{'F1':>8}{'XGB-alone recall':>18}"]
    for c in order:
        h, xa = res_h["per_class"][classes[c]], res_x["per_class"][classes[c]]
        L.append(f"{classes[c]:<26}{h['test_rows']:>10,}{h['precision'] * 100:>10.1f}%{h['recall'] * 100:>8.1f}%"
                 f"{h['f1']:>8.3f}{xa['recall'] * 100:>17.1f}%{'' if h['recall'] >= TARGET else '   < 95%'}")
    report = "\n".join(L)
    print("\n" + report)
    open(os.path.join(HERE, "hybrid_report.txt"), "w", encoding="utf-8").write(report + "\n")

    share = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(12, 10)); ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k), classes, rotation=90, fontsize=7); ax.set_yticks(range(k), classes, fontsize=7)
    for i in range(k):
        for j in range(k):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=5.5,
                        color="white" if share[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Hybrid confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "hybrid_confusion_matrix.png"), dpi=120); plt.close(fig)
    print(f"\nSaved hybrid report, metrics and confusion matrix  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
