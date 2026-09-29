"""Step 4 - Train the stage-2 XGBoost that names the attack.

Trained on the SAME 80% training split as the Isolation Forest (from step 1), so
the 20% test rows are never seen by either model. It learns all 15 classes,
Benign included: when the Isolation Forest raises a false alarm, XGBoost can
still answer "Benign" and cancel it.

Settings match training_xgboost/ (95.7% overall there): depth 10, learning rate
0.1, 500 trees, square-root "balanced" class weights. There is no validation set
(as requested), so it trains all 500 trees; the earlier run peaked at tree 497.

Outputs: xgboost_stage2.json, train_xgboost_report.txt
"""
from if_common import CONFIG_PATH, HERE, XGB_PARAMS, XGB_PATH, load, meta  # keep first (WMI)

import json
import os
import time

import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight


def main():
    t0 = time.time()
    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))      # written by step 2 (same columns)
    keep = np.array(cfg["keep_columns"])
    classes = meta()["classes"]
    y = load("y_train", mmap=False)
    X = np.ascontiguousarray(load("X_train")[:, keep])          # raw values; trees need no transform
    print(f"Training rows {len(y):,}, features {X.shape[1]}, classes {len(classes)}", flush=True)

    w = np.sqrt(compute_sample_weight("balanced", y)).astype(np.float32)   # lifts rare attacks gently
    model = xgb.XGBClassifier(num_class=len(classes), **XGB_PARAMS)
    print("Training XGBoost (500 trees x 15 classes, all CPU cores) ...", flush=True)
    model.fit(X, y, sample_weight=w, verbose=False)
    model.save_model(XGB_PATH)

    pred = model.predict(X)                                     # training fit only; the test is step 5
    lines = ["Stage-2 XGBoost - training split (fit check, not the test result)", "=" * 60,
             f"Rows {len(y):,}   features {X.shape[1]}   trees {XGB_PARAMS['n_estimators']} x {len(classes)} classes",
             f"Training accuracy: {(pred == y).mean():.4f}", ""]
    for c in np.argsort([-(y == k).sum() for k in range(len(classes))]):
        m = y == c
        lines.append(f"{classes[c]:<26}{m.sum():>10,}{(pred[m] == c).mean() * 100:>8.1f}%")
    report = "\n".join(lines)
    print("\n" + report)
    open(os.path.join(HERE, "train_xgboost_report.txt"), "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved {XGB_PATH}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
