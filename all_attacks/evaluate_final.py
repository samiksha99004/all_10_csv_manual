"""Final test of the all-attacks model (with the tuned class weights).

Reuses the exact split from train_all_attacks.prepare_splits(), predicts the 20%
test set (never used for training or tuning), applies the per-class decision
weights from class_weights.json, and saves the final numbers.

Run after train_all_attacks.py and tune_class_weights.py. Outputs (this folder):
  final_metrics.json            overall + per-class precision / recall / F1 / support
  confusion_matrix_final.csv    15x15 counts (rows = actual, columns = predicted)
  confusion_matrix_final.png    the same, coloured by row share
"""
import platform
platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())  # Windows WMI hangs

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from train_all_attacks import HERE, prepare_splits


def save_counts_matrix(cm, classes, path):
    share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(n), classes, fontsize=7)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
    ax.set_title("Confusion matrix, test set (counts; colour = share of the actual class)")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=5.5,
                        color="white" if share[i, j] > 0.5 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    t0 = time.time()
    _, _, X_te, _, _, y_te, classes, features, _ = prepare_splits()
    k = len(classes)

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(HERE, "all_attacks_xgb.json"))
    weights = json.load(open(os.path.join(HERE, "class_weights.json"), encoding="utf-8"))
    w = np.array([weights[c] for c in classes])

    t1 = time.time()
    P = model.predict_proba(X_te[features])
    secs = time.time() - t1
    pred = np.argmax(P * w, axis=1)

    cm = confusion_matrix(y_te, pred, labels=range(k))
    p, r, f, s = precision_recall_fscore_support(y_te, pred, labels=range(k), zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_te, pred)

    metrics = {
        "test_rows": int(len(y_te)),
        "prediction_seconds": round(secs, 2),
        "flows_per_second": int(len(y_te) / secs),
        "accuracy": round(float(acc), 6),
        "macro": {"precision": round(float(pm), 6), "recall": round(float(rm), 6), "f1": round(float(fm), 6)},
        "weighted": {"precision": round(float(pw), 6), "recall": round(float(rw), 6), "f1": round(float(fw), 6)},
        "per_class": {
            classes[i]: {"precision": round(float(p[i]), 6), "recall": round(float(r[i]), 6),
                         "f1": round(float(f[i]), 6), "support": int(s[i]),
                         "true_positive": int(cm[i, i]),
                         "false_negative": int(cm[i].sum() - cm[i, i]),
                         "false_positive": int(cm[:, i].sum() - cm[i, i])}
            for i in range(k)},
    }
    json.dump(metrics, open(os.path.join(HERE, "final_metrics.json"), "w"), indent=2)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(HERE, "confusion_matrix_final.csv"))
    save_counts_matrix(cm, classes, os.path.join(HERE, "confusion_matrix_final.png"))

    print(f"Test rows {len(y_te):,} | accuracy {acc:.4f} | macro F1 {fm:.4f} | weighted F1 {fw:.4f}")
    print(f"Prediction: {secs:.1f}s for {len(y_te):,} flows ({metrics['flows_per_second']:,} flows/s)")
    for i in np.argsort(-s):
        print(f"  {classes[i]:<26} P {p[i]:.4f}  R {r[i]:.4f}  F1 {f[i]:.4f}  n {s[i]:>7,}")
    print(f"Saved final_metrics.json, confusion_matrix_final.csv/.png  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
