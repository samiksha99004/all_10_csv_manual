"""Step 2 - train the Isolation Forest and score every row (train and test).

Fitted on the Benign rows of the 80% training split only (features log1p-compressed and
standardised, scaler fitted on Benign training rows - same method as the earlier hybrids).

UNLIKE the earlier hybrids, this one does NOT use a threshold to gate which flows reach
XGBoost. Instead every row's anomaly score is saved, so step 3 can attach it to XGBoost's
inputs as one more column. A threshold and ROC-AUC are still reported in train_report.txt,
for information only (how separable normal/attack is on this score alone).

Outputs: isolation_forest.joblib, scaler.joblib, if_config.json, train_report.txt,
         data/train_scores.npy, data/test_scores.npy
"""
from h2_common import (CHUNK_ROWS, DATA, HERE, IF_CONTAMINATION, IF_MAX_SAMPLES, IF_N_ESTIMATORS,
                       SEED, anomaly_scores, load, log_features, meta)  # keep first (WMI)

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
    X_train, y_train = load("X_train"), load("y_train", mmap=False)
    X_test, y_test = load("X_test"), load("y_test", mmap=False)

    # drop columns that never change in the training data (carry no split information)
    lo = np.full(X_train.shape[1], np.inf, np.float32); hi = np.full(X_train.shape[1], -np.inf, np.float32)
    for s in range(0, len(X_train), CHUNK_ROWS):
        part = X_train[s:s + CHUNK_ROWS]; lo = np.minimum(lo, part.min(0)); hi = np.maximum(hi, part.max(0))
    keep = np.flatnonzero(hi > lo)
    print(f"using {len(keep)} of {X_train.shape[1]} features for the Isolation Forest", flush=True)

    benign_idx = np.flatnonzero(y_train == benign)
    scaler = StandardScaler().fit(log_features(X_train[benign_idx], keep))
    joblib.dump(scaler, os.path.join(HERE, "scaler.joblib"))

    Xb = scaler.transform(log_features(X_train[benign_idx], keep)).astype(np.float32)
    print(f"fitting Isolation Forest on {len(Xb):,} Benign rows "
          f"({IF_N_ESTIMATORS} trees, {IF_MAX_SAMPLES} rows/tree) ...", flush=True)
    model = IsolationForest(n_estimators=IF_N_ESTIMATORS, max_samples=IF_MAX_SAMPLES,
                            contamination=IF_CONTAMINATION, n_jobs=-1, random_state=SEED).fit(Xb)
    del Xb
    joblib.dump(model, os.path.join(HERE, "isolation_forest.joblib"))

    print("scoring every training and test row (these scores become an XGBoost input column) ...", flush=True)
    train_scores = anomaly_scores(model, scaler, X_train, keep)
    test_scores = anomaly_scores(model, scaler, X_test, keep)
    np.save(os.path.join(DATA, "train_scores.npy"), train_scores)
    np.save(os.path.join(DATA, "test_scores.npy"), test_scores)

    # threshold + ROC-AUC: reported for information only, not used to route anything
    y_attack = (y_train != benign)
    auc = float(roc_auc_score(y_attack, train_scores))
    ben_sorted = np.sort(train_scores[benign_idx]); atk_sorted = np.sort(train_scores[y_attack])
    cand = np.unique(np.quantile(train_scores, np.linspace(0.005, 0.995, 3000)))
    best_j, threshold = -1.0, float(cand[0])
    for t in cand:
        normal_acc = np.searchsorted(ben_sorted, t, "left") / len(ben_sorted)
        attack_rec = (len(atk_sorted) - np.searchsorted(atk_sorted, t, "left")) / len(atk_sorted)
        j = normal_acc + attack_rec - 1.0
        if j > best_j:
            best_j, threshold = j, float(t)

    json.dump({"keep_columns": keep.tolist(), "features_used": [m["features"][i] for i in keep],
               "classes": classes, "benign_code": benign, "n_estimators": IF_N_ESTIMATORS,
               "max_samples": IF_MAX_SAMPLES, "contamination": IF_CONTAMINATION,
               "transform": "log1p(x + 1) then StandardScaler",
               "score": "-score_samples (higher = more unusual); fed to XGBoost as an extra column, not used as a gate",
               "informational_threshold": threshold, "informational_roc_auc": auc},
              open(os.path.join(HERE, "if_config.json"), "w", encoding="utf-8"), indent=2)

    flagged = train_scores >= threshold
    benign_acc = 1 - flagged[benign_idx].mean(); attack_rec = flagged[y_attack].mean()
    lines = [f"Isolation Forest (retraining_hybrid_2) - training split ({len(y_train):,} rows)", "=" * 60,
             f"Trees {IF_N_ESTIMATORS}, rows/tree {IF_MAX_SAMPLES}, contamination {IF_CONTAMINATION}, "
             f"features {len(keep)}",
             f"ROC-AUC {auc:.3f} (how separable normal/attack is on the score alone)",
             f"Informational threshold {threshold:.5f} (max balanced accuracy) - NOT used to gate XGBoost's input",
             f"At that threshold: normal accuracy {benign_acc*100:.1f}%  attack recall {attack_rec*100:.1f}%", ""]
    for c in np.argsort([-(y_train == k).sum() for k in range(len(classes))]):
        mk = y_train == c
        r = 1 - flagged[mk].mean() if c == benign else flagged[mk].mean()
        lines.append(f"{classes[c]:<26}{mk.sum():>10,}{r * 100:>8.1f}%")
    report = "\n".join(lines)
    print("\n" + report)
    open(os.path.join(HERE, "train_report.txt"), "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved model, scaler, scores ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
