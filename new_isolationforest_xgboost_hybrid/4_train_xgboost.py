"""Step 4 - train the final XGBoost classifier on the stacked features.

Input: the 63 original features + the Isolation Forest score (64 columns total, built by
step 3). This is the ONLY classifier in this version - there is no separate gate. It
predicts all 15 original classes directly (Benign + 14 attack types, kept separate,
nothing merged).

Output: xgboost_model.json, train_report.txt
"""
from h2_common import DATA, HERE, XGB_PARAMS, load  # keep first (WMI)

import json
import os
import time

import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight


def main():
    t0 = time.time()
    m = json.load(open(os.path.join(DATA, "meta_stacked.json"), encoding="utf-8"))
    classes = m["classes"]
    y = load("y_train", mmap=False)
    X = load("X_train_stacked", mmap=False)
    print(f"training rows {len(y):,}, features {X.shape[1]} (63 + Isolation Forest score), "
          f"classes {len(classes)}", flush=True)

    w = np.sqrt(compute_sample_weight("balanced", y)).astype(np.float32)   # lifts rare attacks gently
    model = xgb.XGBClassifier(num_class=len(classes), **XGB_PARAMS)
    print(f"training XGBoost ({XGB_PARAMS['n_estimators']} trees x {len(classes)} classes) ...", flush=True)
    model.fit(X, y, sample_weight=w, verbose=False)
    model.save_model(os.path.join(HERE, "xgboost_model.json"))

    pred = model.predict(X)
    lines = ["XGBoost on stacked features (retraining_hybrid_2) - training split (fit check, not the test result)",
             "=" * 70,
             f"Rows {len(y):,}   features {X.shape[1]}   trees {XGB_PARAMS['n_estimators']} x {len(classes)} classes",
             f"Training accuracy: {(pred == y).mean():.4f}", ""]
    for c in np.argsort([-(y == k).sum() for k in range(len(classes))]):
        mk = y == c
        lines.append(f"{classes[c]:<26}{mk.sum():>10,}{(pred[mk] == c).mean() * 100:>8.1f}%")
    report = "\n".join(lines)
    print("\n" + report)
    open(os.path.join(HERE, "train_xgboost_report.txt"), "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved xgboost_model.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
