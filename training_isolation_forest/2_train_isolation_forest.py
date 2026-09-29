"""Step 2 - Train the Isolation Forest and choose the alarm threshold.

Isolation Forest is unsupervised: it learns what normal traffic looks like and
flags anything unusual. It is fitted on the Benign rows of the 80% training
split only (novelty detection), because large attack classes such as
DDoS-LOIC-HTTP (460k training rows) would otherwise be learned as "normal".

The alarm threshold is then chosen on the whole training split (Benign +
attacks). The goal is as many classes as possible at >= 95% (attacks caught;
for Benign, flows NOT flagged), with overall accuracy as the tie-breaker. The
test set is not touched here.

Outputs: isolation_forest.joblib, model_config.json, train_report.txt,
         data/train_scores.npy
"""
from if_common import (CONFIG_PATH, DATA, MAX_FEATURES, MAX_SAMPLES, MODEL_PATH, N_ESTIMATORS, N_JOBS,
                       SEED, TARGET, HERE, anomaly_scores, load, meta, transform)  # keep first (WMI)

import json
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


def choose_threshold(scores, y, benign, n_classes):
    """Pick the threshold with the most classes >= 95%, then the best overall accuracy."""
    sorted_by_class = {c: np.sort(scores[y == c]) for c in range(n_classes)}
    candidates = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 2000)))
    n_total = len(y)
    best, best_t = None, None
    for t in candidates:
        rates, correct = [], 0
        for c, s in sorted_by_class.items():
            flagged = len(s) - np.searchsorted(s, t, side="left")      # scores >= t
            ok = (len(s) - flagged) if c == benign else flagged
            correct += ok
            rates.append(ok / len(s))
        score = (int(sum(r >= TARGET for r in rates)), correct / n_total)
        if best is None or score > best:
            best, best_t = score, float(t)
    return best_t, best


def main():
    t0 = time.time()
    m = meta()
    classes, benign = m["classes"], m["benign_code"]
    X, y = load("X_train"), load("y_train", mmap=False)

    # columns that never change carry no information for splitting (the 8 all-zero ones)
    lo = np.full(X.shape[1], np.inf, np.float32); hi = np.full(X.shape[1], -np.inf, np.float32)
    for s in range(0, len(X), 200_000):
        part = X[s:s + 200_000]
        lo = np.minimum(lo, part.min(0)); hi = np.maximum(hi, part.max(0))
    keep = np.flatnonzero(hi > lo)
    print(f"Using {len(keep)} of {X.shape[1]} features (dropped constant columns)", flush=True)

    Xb = transform(X[np.flatnonzero(y == benign)], keep)
    print(f"Fitting Isolation Forest on {len(Xb):,} Benign training rows ...", flush=True)
    model = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES, max_features=MAX_FEATURES,
                            contamination="auto", n_jobs=N_JOBS, random_state=SEED).fit(Xb)
    del Xb

    print(f"Scoring all {len(X):,} training rows in chunks ...", flush=True)
    scores = anomaly_scores(model, X, keep)
    np.save(os.path.join(DATA, "train_scores.npy"), scores)

    threshold, (n_ok, acc) = choose_threshold(scores, y, benign, len(classes))
    joblib.dump(model, MODEL_PATH)
    json.dump({"threshold": threshold, "keep_columns": keep.tolist(),
               "features_used": [m["features"][i] for i in keep], "classes": classes, "benign_code": benign,
               "n_estimators": N_ESTIMATORS, "max_samples": MAX_SAMPLES, "max_features": MAX_FEATURES,
               "transform": "log1p(x + 1)", "score": "-score_samples (higher = more unusual)"},
              open(CONFIG_PATH, "w"), indent=2)

    flagged = scores >= threshold
    lines = [f"Isolation Forest - training split ({len(y):,} rows)", "=" * 60,
             f"Trees {N_ESTIMATORS}, rows per tree {MAX_SAMPLES}, features {len(keep)}",
             f"Chosen threshold (anomaly score): {threshold:.5f}",
             f"Classes at >= 95%: {n_ok}/{len(classes)}   overall accuracy: {acc:.4f}", "",
             f"{'class':<26}{'rows':>10}{'rate':>9}   (attacks: caught; Benign: not flagged)"]
    for c in np.argsort([-(y == k).sum() for k in range(len(classes))]):
        mk = y == c
        r = 1 - flagged[mk].mean() if c == benign else flagged[mk].mean()
        lines.append(f"{classes[c]:<26}{mk.sum():>10,}{r * 100:>8.1f}%{'' if r >= TARGET else '   < 95%'}")
    report = "\n".join(lines)
    print("\n" + report)
    open(os.path.join(HERE, "train_report.txt"), "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved model and threshold  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
