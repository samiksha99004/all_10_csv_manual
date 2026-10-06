"""Step 2 - train the Isolation Forest (stage 1) and set the alarm threshold.

Fitted on the Benign rows of the 80% training split only. Features are log1p-compressed
and standardised (scaler fitted on Benign training rows). Threshold = point of maximum
balanced accuracy (Youden's J) between normal and attack.

Outputs: isolation_forest.joblib, scaler.joblib, model_config.json, train_report.txt
"""
from hy_common import (CONTAMINATION, DATA, HERE, MAX_SAMPLES, N_ESTIMATORS, SEED, TARGET,
                       anomaly_scores, load, log_features, meta)  # keep first (WMI)

import json
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def main():
    t0 = time.time()
    m = meta()
    classes, benign = m["classes"], m["benign_code"]
    X, y = load("X_train"), load("y_train", mmap=False)

    # drop columns that never change (carry no split information)
    lo = np.full(X.shape[1], np.inf, np.float32); hi = np.full(X.shape[1], -np.inf, np.float32)
    for s in range(0, len(X), 200_000):
        part = X[s:s + 200_000]; lo = np.minimum(lo, part.min(0)); hi = np.maximum(hi, part.max(0))
    keep = np.flatnonzero(hi > lo)
    print(f"using {len(keep)} of {X.shape[1]} features", flush=True)

    benign_idx = np.flatnonzero(y == benign)
    scaler = StandardScaler().fit(log_features(X[benign_idx], keep))
    joblib.dump(scaler, os.path.join(HERE, "scaler.joblib"))

    Xb = scaler.transform(log_features(X[benign_idx], keep)).astype(np.float32)
    print(f"fitting Isolation Forest on {len(Xb):,} Benign rows "
          f"({N_ESTIMATORS} trees, {MAX_SAMPLES} rows/tree) ...", flush=True)
    model = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES, contamination=CONTAMINATION,
                            n_jobs=-1, random_state=SEED).fit(Xb)
    del Xb

    print(f"scoring all {len(X):,} training rows ...", flush=True)
    scores = anomaly_scores(model, X, keep, scaler)

    y_attack = (y != benign)
    auc = float(roc_auc_score(y_attack, scores))
    ben_sorted = np.sort(scores[benign_idx]); atk_sorted = np.sort(scores[y_attack])
    cand = np.unique(np.quantile(scores, np.linspace(0.005, 0.995, 3000)))
    best_j, threshold = -1.0, float(cand[0])
    for t in cand:
        normal_acc = np.searchsorted(ben_sorted, t, "left") / len(ben_sorted)
        attack_rec = (len(atk_sorted) - np.searchsorted(atk_sorted, t, "left")) / len(atk_sorted)
        j = normal_acc + attack_rec - 1.0
        if j > best_j:
            best_j, threshold = j, float(t)
    joblib.dump(model, os.path.join(HERE, "isolation_forest.joblib"))
    json.dump({"threshold": threshold, "roc_auc": auc, "keep_columns": keep.tolist(),
               "features_used": [m["features"][i] for i in keep], "classes": classes, "benign_code": benign,
               "n_estimators": N_ESTIMATORS, "max_samples": MAX_SAMPLES, "contamination": CONTAMINATION,
               "transform": "log1p(x + 1) then StandardScaler", "score": "-score_samples (higher = more unusual)"},
              open(os.path.join(HERE, "model_config.json"), "w", encoding="utf-8"), indent=2)

    flagged = scores >= threshold
    benign_acc = 1 - flagged[benign_idx].mean()
    attack_rec = flagged[y_attack].mean()
    lines = [f"Isolation Forest (retrain, cleaned_dataset_2) - training split ({len(y):,} rows)", "=" * 60,
             f"Trees {N_ESTIMATORS}, rows/tree {MAX_SAMPLES}, contamination {CONTAMINATION}, features {len(keep)}",
             f"Threshold {threshold:.5f} (max balanced accuracy). ROC-AUC {auc:.3f}",
             f"Normal accuracy: {benign_acc * 100:.1f}%   Attack recall: {attack_rec * 100:.1f}%   "
             f"Balanced accuracy: {(benign_acc + attack_rec) / 2 * 100:.1f}%", ""]
    n_ok = 0
    for c in np.argsort([-(y == k).sum() for k in range(len(classes))]):
        mk = y == c
        r = 1 - flagged[mk].mean() if c == benign else flagged[mk].mean()
        n_ok += r >= TARGET
        lines.append(f"{classes[c]:<26}{mk.sum():>10,}{r * 100:>8.1f}%{'' if r >= TARGET else '   < 95%'}")
    lines.insert(5, f"Classes at >= 95%: {n_ok}/{len(classes)}")
    report = "\n".join(lines)
    print("\n" + report)
    open(os.path.join(HERE, "train_report.txt"), "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved model, scaler and threshold ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
