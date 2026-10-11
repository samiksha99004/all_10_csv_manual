"""Step 2 - Train the Isolation Forest and set the alarm threshold.

Isolation Forest learns what normal traffic looks like and flags anything
unusual. It is fitted on the Benign rows of the 80% training split only.

Fixes over the first version (which flagged 95% of normal traffic):
  - features are log1p-compressed AND standardised (scaler fitted on the Benign
    training rows), so no single column dominates the splits;
  - contamination is low (2%);
  - the threshold is the point that best separates normal from attack (maximum
    balanced accuracy = Youden's J), so stage-1 accuracy is as high as this
    Isolation Forest allows. ROC-AUC is also reported (threshold-free).

Outputs: isolation_forest.joblib, scaler.joblib, model_config.json,
         train_report.txt, data/train_scores.npy
"""
from if_common import (CONFIG_PATH, CONTAMINATION, DATA, HERE, MAX_FEATURES, MAX_SAMPLES, MODEL_PATH,
                       N_ESTIMATORS, N_JOBS, SCALER_PATH, SEED, TARGET, anomaly_scores, load,
                       log_features, meta)  # keep first (WMI)

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

    # drop columns that never change (the 8 all-zero ones carry no split information)
    lo = np.full(X.shape[1], np.inf, np.float32); hi = np.full(X.shape[1], -np.inf, np.float32)
    for s in range(0, len(X), 200_000):
        part = X[s:s + 200_000]; lo = np.minimum(lo, part.min(0)); hi = np.maximum(hi, part.max(0))
    keep = np.flatnonzero(hi > lo)
    print(f"Using {len(keep)} of {X.shape[1]} features", flush=True)

    benign_idx = np.flatnonzero(y == benign)
    scaler = StandardScaler().fit(log_features(X[benign_idx], keep))    # fit on normal traffic only
    joblib.dump(scaler, SCALER_PATH)

    Xb = scaler.transform(log_features(X[benign_idx], keep)).astype(np.float32)
    print(f"Fitting Isolation Forest on {len(Xb):,} Benign rows "
          f"({N_ESTIMATORS} trees, {MAX_SAMPLES} rows/tree) ...", flush=True)
    model = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES, max_features=MAX_FEATURES,
                            contamination=CONTAMINATION, n_jobs=N_JOBS, random_state=SEED).fit(Xb)
    del Xb

    print(f"Scoring all {len(X):,} training rows in chunks ...", flush=True)
    scores = anomaly_scores(model, X, keep, scaler)
    np.save(os.path.join(DATA, "train_scores.npy"), scores)

    # Threshold that best separates normal vs attack (max balanced accuracy = Youden's J).
    y_attack = (y != benign)
    auc = float(roc_auc_score(y_attack, scores))
    ben_sorted = np.sort(scores[benign_idx]); atk_sorted = np.sort(scores[y_attack])
    cand = np.unique(np.quantile(scores, np.linspace(0.005, 0.995, 3000)))
    best_j, threshold = -1.0, float(cand[0])
    for t in cand:
        normal_acc = np.searchsorted(ben_sorted, t, "left") / len(ben_sorted)          # benign score < t
        attack_rec = (len(atk_sorted) - np.searchsorted(atk_sorted, t, "left")) / len(atk_sorted)
        j = normal_acc + attack_rec - 1.0
        if j > best_j:
            best_j, threshold = j, float(t)
    joblib.dump(model, MODEL_PATH)
    json.dump({"threshold": threshold, "roc_auc": auc, "keep_columns": keep.tolist(),
               "features_used": [m["features"][i] for i in keep], "classes": classes, "benign_code": benign,
               "n_estimators": N_ESTIMATORS, "max_samples": MAX_SAMPLES, "contamination": CONTAMINATION,
               "transform": "log1p(x + 1) then StandardScaler", "score": "-score_samples (higher = more unusual)"},
              open(CONFIG_PATH, "w"), indent=2)

    flagged = scores >= threshold
    benign_acc = 1 - flagged[benign_idx].mean()
    attack_rec = flagged[y != benign].mean()
    n_ok = 0
    lines = [f"Isolation Forest - training split ({len(y):,} rows)", "=" * 60,
             f"Trees {N_ESTIMATORS}, rows/tree {MAX_SAMPLES}, contamination {CONTAMINATION}, features {len(keep)}",
             f"Standardised features. Threshold {threshold:.5f} (max balanced accuracy). ROC-AUC {auc:.3f}",
             f"Normal accuracy: {benign_acc * 100:.1f}%   Attack recall: {attack_rec * 100:.1f}%   "
             f"Balanced accuracy: {(benign_acc + attack_rec) / 2 * 100:.1f}%", "",
             f"{'class':<26}{'rows':>10}{'rate':>9}   (attacks: caught; Benign: not flagged)"]
    for c in np.argsort([-(y == k).sum() for k in range(len(classes))]):
        mk = y == c
        r = 1 - flagged[mk].mean() if c == benign else flagged[mk].mean()
        n_ok += r >= TARGET
        lines.append(f"{classes[c]:<26}{mk.sum():>10,}{r * 100:>8.1f}%{'' if r >= TARGET else '   < 95%'}")
    lines.insert(5, f"Classes at >= 95%: {n_ok}/{len(classes)}")
    report = "\n".join(lines)
    print("\n" + report)
    open(os.path.join(HERE, "train_report.txt"), "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved model, scaler and threshold  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
