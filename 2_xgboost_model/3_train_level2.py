"""Step 3 - Level 2: multi-class XGBoost naming the attack (14 attack classes).

Trained on the ATTACK rows of the training split only (Benign excluded), because
level 1 already removes Benign. Labels are re-indexed to 0..13 over the attacks;
attack_classes.json maps them back to names.

Output: level2_multiclass_xgb.json, attack_classes.json, level2_report.txt
"""
from xgb_common import DATA, HERE, L2_PATH, N_ESTIMATORS, PARAMS, load, meta  # keep first (WMI)

import json
import os
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight


def main():
    t0 = time.time()
    m = meta(); classes, benign = m["classes"], m["benign_code"]
    X, yc = load("X_train"), load("y_train", mmap=False)
    mask = yc != benign
    Xa = np.ascontiguousarray(X[mask]); yc_a = yc[mask]

    attack_codes = sorted(int(c) for c in np.unique(yc_a))          # original codes of the 14 attacks
    remap = {c: i for i, c in enumerate(attack_codes)}
    attack_classes = [classes[c] for c in attack_codes]            # attack_classes[i] = name
    y = np.array([remap[int(c)] for c in yc_a], dtype=np.int16)
    json.dump({str(i): name for i, name in enumerate(attack_classes)},
              open(os.path.join(DATA, "attack_classes.json"), "w"), indent=2)

    print(f"Training level-2 (which attack) on {len(y):,} attack rows, {len(attack_classes)} classes ...", flush=True)
    w = np.sqrt(compute_sample_weight("balanced", y)).astype(np.float32)
    model = xgb.XGBClassifier(objective="multi:softprob", num_class=len(attack_classes),
                              n_estimators=N_ESTIMATORS, **PARAMS)
    model.fit(Xa, y, sample_weight=w, verbose=False)
    model.save_model(L2_PATH)

    pred = model.predict(Xa)
    lines = ["Level 2 - which attack (training fit)", "=" * 45,
             f"Rows {len(y):,}   classes {len(attack_classes)}",
             f"Training accuracy: {accuracy_score(y, pred):.4f}", ""]
    for i in np.argsort([-(y == k).sum() for k in range(len(attack_classes))]):
        mk = y == i
        lines.append(f"{attack_classes[i]:<26}{mk.sum():>10,}{(pred[mk] == i).mean()*100:>8.1f}%")
    report = "\n".join(lines)
    print(report)
    open(os.path.join(HERE, "level2_report.txt"), "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved {L2_PATH}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
