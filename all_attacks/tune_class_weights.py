"""Tune per-class decision weights for the multi-class model (no retraining).

The trained model outputs a probability for each of the 15 classes; normally the
class with the highest probability wins. Rare classes often come second to
Benign by a small margin. This step multiplies each class's probability by a
weight before picking the winner:

    predicted class = argmax( probability[c] * weight[c] )

The weights are searched on the VALIDATION set only (coordinate ascent over a
log-spaced grid). The goal is to get as many classes as possible to >= 95%
recall, then maximise mean recall, while keeping Benign recall and overall
accuracy >= 95%. The TEST set is untouched until the final score, so the
reported numbers are not tuned on test data.

Run after train_all_attacks.py. Outputs (this folder):
  class_weights.json               the 15 weights, by class name
  evaluation_report_tuned.txt      test results before vs after tuning
  per_class_recall_tuned.png       recall bar per class with the 95% line
"""
import platform
platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())  # Windows WMI hangs

import json
import os
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report

from train_all_attacks import HERE, prepare_splits, save_recall_bar

GRID = 2.0 ** np.arange(-3, 8.5, 0.5)   # weights 0.125 .. 256
PASSES = 4
TARGET = 0.95


def per_class_recall(y, pred, k):
    hits = np.bincount(y[pred == y], minlength=k)
    support = np.bincount(y, minlength=k)
    return hits / np.maximum(support, 1)


def evaluate(P, y, w, benign):
    pred = np.argmax(P * w, axis=1)
    rec = per_class_recall(y, pred, P.shape[1])
    acc = float((pred == y).mean())
    feasible = rec[benign] >= TARGET and acc >= TARGET
    return (feasible, int((rec >= TARGET).sum()), float(rec.mean())), rec, acc


def main():
    t0 = time.time()
    _, X_val, X_te, _, y_val, y_te, classes, features, _ = prepare_splits()
    benign = classes.index("Benign")

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(HERE, "all_attacks_xgb.json"))
    print("Predicting probabilities ...", flush=True)
    P_val = model.predict_proba(X_val[features])
    P_te = model.predict_proba(X_te[features])
    del X_val, X_te

    k = len(classes)
    w = np.ones(k)
    best, _, _ = evaluate(P_val, y_val, w, benign)
    print(f"start (validation): feasible={best[0]} classes>=95%={best[1]} mean recall={best[2]:.4f}", flush=True)
    for p in range(PASSES):
        for c in range(k):
            for g in GRID:
                trial = w.copy()
                trial[c] = g
                score, _, _ = evaluate(P_val, y_val, trial, benign)
                if score > best:
                    best, w = score, trial
        print(f"pass {p+1}: feasible={best[0]} classes>=95%={best[1]} mean recall={best[2]:.4f}", flush=True)

    # final, honest score on the untouched test set
    _, rec0, acc0 = evaluate(P_te, y_te, np.ones(k), benign)
    _, rec1, acc1 = evaluate(P_te, y_te, w, benign)
    pred1 = np.argmax(P_te * w, axis=1)

    json.dump({c: round(float(w[i]), 4) for i, c in enumerate(classes)},
              open(os.path.join(HERE, "class_weights.json"), "w"), indent=2)
    save_recall_bar(rec1, classes, os.path.join(HERE, "per_class_recall_tuned.png"))

    support = np.bincount(y_te, minlength=k)
    lines = [
        "Multi-class model with tuned per-class decision weights (test set)",
        "=" * 72,
        "Weights searched on the validation set only; the test set was not used for tuning.",
        "",
        f"Overall accuracy:   before {acc0:.4f}   after {acc1:.4f}",
        f"Mean recall:        before {rec0.mean():.4f}   after {rec1.mean():.4f}",
        f"Classes >= 95%:     before {int((rec0 >= TARGET).sum())}/{k}   after {int((rec1 >= TARGET).sum())}/{k}",
        "",
        f"{'class':<26}{'test rows':>10}{'weight':>9}{'recall before':>15}{'recall after':>14}",
    ]
    for i in np.argsort(-support):
        flag = "" if rec1[i] >= TARGET else "   < 95%"
        lines.append(f"{classes[i]:<26}{support[i]:>10,}{w[i]:>9.3g}{rec0[i]*100:>14.1f}%{rec1[i]*100:>13.1f}%{flag}")
    lines += ["", "Full report after tuning:", ""]
    lines.append(classification_report(y_te, pred1, labels=range(k), target_names=classes,
                                        digits=4, zero_division=0))
    report = "\n".join(lines)
    print("\n" + report, flush=True)
    open(os.path.join(HERE, "evaluation_report_tuned.txt"), "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved class_weights.json, report and plot  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
