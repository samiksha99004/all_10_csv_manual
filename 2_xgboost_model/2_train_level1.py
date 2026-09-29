"""Step 2 - Level 1: binary XGBoost, attack (1) vs not-attack / Benign (0).

Trained on the whole training split. Every non-Benign label becomes 1.
Balanced sample weights handle the Benign majority.

Output: level1_binary_xgb.json, level1_report.txt
"""
from xgb_common import HERE, L1_PATH, N_ESTIMATORS, PARAMS, load, meta  # keep first (WMI)

import os
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight


def main():
    t0 = time.time()
    benign = meta()["benign_code"]
    X, yc = load("X_train"), load("y_train", mmap=False)
    y = (yc != benign).astype(np.int8)                    # 1 = attack, 0 = Benign
    print(f"Training level-1 (attack vs not) on {len(y):,} rows ...", flush=True)
    w = compute_sample_weight("balanced", y)
    model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=N_ESTIMATORS,
                              eval_metric="logloss", scale_pos_weight=1, **PARAMS)
    model.fit(np.ascontiguousarray(X), y, sample_weight=w, verbose=False)
    model.save_model(L1_PATH)

    pred = model.predict(np.ascontiguousarray(X))
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    report = ("Level 1 - attack vs not (training fit)\n" + "=" * 45 +
              f"\nRows {len(y):,}\nTraining accuracy: {accuracy_score(y, pred):.4f}\n"
              f"Attack: precision {p:.4f}  recall {r:.4f}  F1 {f:.4f}\n")
    print(report)
    open(os.path.join(HERE, "level1_report.txt"), "w", encoding="utf-8").write(report)
    print(f"Saved {L1_PATH}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
