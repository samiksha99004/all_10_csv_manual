"""Step 4 - test the hybrid (stage 1 -> stage 2) on the 20% test split (unseen by both).

Flows stage 1 flags go to stage 2, which names the attack or answers Benign (cancels the
alarm). Flows stage 1 calls normal are final Benign. Reports overall accuracy, per-class
precision/recall/F1, and the confusion matrix.

Outputs: hybrid_metrics.json, hybrid_confusion_matrix.csv
"""
from hy_common import DATA, HERE, load, meta  # keep first (WMI)

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def main():
    t0 = time.time()
    m = meta()
    classes, benign = m["classes"], m["benign_code"]
    cfg = json.load(open(os.path.join(HERE, "model_config.json"), encoding="utf-8"))
    keep = np.array(cfg["keep_columns"])
    scaler = joblib.load(os.path.join(HERE, "scaler.joblib"))
    iforest = joblib.load(os.path.join(HERE, "isolation_forest.joblib"))
    s2 = xgb.Booster(); s2.load_model(os.path.join(HERE, "xgboost_stage2.json"))

    X, y = load("X_test"), load("y_test", mmap=False)
    print(f"scoring {len(X):,} test rows ...", flush=True)
    Xk = np.ascontiguousarray(X[:, keep])
    score = -iforest.score_samples(scaler.transform(np.log1p(np.clip(Xk, 0, None) + 1.0)))
    flagged = score >= cfg["threshold"]
    pred = np.full(len(X), benign, dtype=np.int64)
    if flagged.any():
        p = s2.predict(xgb.DMatrix(Xk[flagged]))
        pred[flagged] = p.argmax(1)

    acc = float((pred == y).mean())
    k = len(classes)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(k), zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(k))
    per = {classes[c]: {"test_rows": int(s[c]), "precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c])}
           for c in range(k)}
    n_ok = sum(v["recall"] >= 0.95 for v in per.values())

    json.dump({"overall_accuracy": acc, "classes_meeting_95": n_ok, "stage1_flagged_pct": float(flagged.mean()),
               "per_class": per}, open(os.path.join(HERE, "hybrid_metrics.json"), "w", encoding="utf-8"), indent=2)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(HERE, "hybrid_confusion_matrix.csv"))

    order = sorted(range(k), key=lambda c: -s[c])
    lines = [f"Hybrid (retrain, cleaned_dataset_2) - test split ({len(y):,} rows)", "=" * 70,
             f"Overall accuracy: {acc:.4f}   stage-1 flagged: {flagged.mean()*100:.1f}%   "
             f"classes >= 95% recall: {n_ok}/{k}", "",
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
