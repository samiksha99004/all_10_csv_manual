"""Step 5 - test the stacked model on the 20% test split (never used in training).

Both the Isolation Forest and XGBoost were fitted using only the 80% training split, so
these test rows are unseen by both. Reports overall accuracy, per-class precision/recall/F1,
and the confusion matrix, for the real 15 separate classes.

Outputs: metrics.json, confusion_matrix.csv
"""
from h2_common import DATA, HERE, load  # keep first (WMI)

import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def main():
    t0 = time.time()
    m = json.load(open(os.path.join(DATA, "meta_stacked.json"), encoding="utf-8"))
    classes = m["classes"]
    k = len(classes)
    X = load("X_test_stacked", mmap=False)
    y = load("y_test", mmap=False)

    model = xgb.XGBClassifier(); model.load_model(os.path.join(HERE, "xgboost_model.json"))
    pred = model.predict(X)

    acc = float((pred == y).mean())
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(k), zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(k))
    per = {classes[c]: {"test_rows": int(s[c]), "precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c])}
           for c in range(k)}
    n_ok = sum(v["recall"] >= 0.95 for v in per.values())

    json.dump({"overall_accuracy": acc, "classes_meeting_95": n_ok, "per_class": per},
              open(os.path.join(HERE, "metrics.json"), "w", encoding="utf-8"), indent=2)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(HERE, "confusion_matrix.csv"))

    order = sorted(range(k), key=lambda c: -s[c])
    lines = [f"Stacked Isolation Forest + XGBoost (retraining_hybrid_2) - test split ({len(y):,} rows)",
             "=" * 70, f"Overall accuracy: {acc:.4f}   classes >= 95% recall: {n_ok}/{k}", "",
             f"{'class':<26}{'test rows':>10}{'precision':>11}{'recall':>9}{'F1':>8}"]
    for c in order:
        v = per[classes[c]]
        lines.append(f"{classes[c]:<26}{v['test_rows']:>10,}{v['precision']*100:>10.1f}%{v['recall']*100:>8.1f}%"
                     f"{v['f1']:>8.3f}{'' if v['recall'] >= 0.95 else '   < 95%'}")
    report = "\n".join(lines)
    print("\n" + report)
    print(f"\ndone ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
