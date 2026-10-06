"""Step 3 - train the stage-2 XGBoost (names the attack, Benign included).

Trained on the SAME 80% training split as the Isolation Forest (step 2), so the 20% test
rows are never seen by either model. 200 trees (tree size requested), depth 10, learning
rate 0.1, square-root "balanced" sample weights, all classes including Benign so XGBoost
can cancel a stage-1 false alarm by answering Benign.

Output: xgboost_stage2.json, train_xgboost_report.txt
"""
from hy_common import HERE, XGB_PARAMS, load, meta  # keep first (WMI)

import json
import os
import time

import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight


def main():
    t0 = time.time()
    cfg = json.load(open(os.path.join(HERE, "model_config.json"), encoding="utf-8"))  # written by step 2
    keep = np.array(cfg["keep_columns"])
    classes = meta()["classes"]
    y = load("y_train", mmap=False)
    X = np.ascontiguousarray(load("X_train")[:, keep])
    print(f"training rows {len(y):,}, features {X.shape[1]}, classes {len(classes)}", flush=True)

    w = np.sqrt(compute_sample_weight("balanced", y)).astype(np.float32)
    model = xgb.XGBClassifier(num_class=len(classes), **XGB_PARAMS)
    print(f"training XGBoost ({XGB_PARAMS['n_estimators']} trees x {len(classes)} classes) ...", flush=True)
    model.fit(X, y, sample_weight=w, verbose=False)
    model.save_model(os.path.join(HERE, "xgboost_stage2.json"))

    pred = model.predict(X)
    lines = ["Stage-2 XGBoost (retrain, cleaned_dataset_2) - training split (fit check, not the test result)",
             "=" * 60,
             f"Rows {len(y):,}   features {X.shape[1]}   trees {XGB_PARAMS['n_estimators']} x {len(classes)} classes",
             f"Training accuracy: {(pred == y).mean():.4f}", ""]
    for c in np.argsort([-(y == k).sum() for k in range(len(classes))]):
        m = y == c
        lines.append(f"{classes[c]:<26}{m.sum():>10,}{(pred[m] == c).mean() * 100:>8.1f}%")
    report = "\n".join(lines)
    print("\n" + report)
    open(os.path.join(HERE, "train_xgboost_report.txt"), "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved xgboost_stage2.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
