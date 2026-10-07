"""Shared settings for retraining_hybrid_2 (stacked Isolation Forest + XGBoost, 15 separate classes).
Import this first (WMI patch for Windows)."""
import platform
platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
# 64 columns: 63 features + Label. The 15 unnecessary columns (8 all-zero, 7 exact duplicates
# of another column) are already removed from this file. See CLAUDE.md / chat for the list.
CLEAN_CSV = r"F:\all_10_csv_manual\original_csv\cleaned_dataset_2.csv"
SUMMARY = r"F:\all_10_csv_manual\cleaning_dataset\9_final_summary.json"

BENIGN_SAMPLE = 1_000_000
CHUNK_ROWS = 200_000
TEST_SIZE = 0.20
SEED = 42

# Isolation Forest (its score becomes an extra XGBoost feature; no gating, no threshold used
# for routing). 300 trees / 4096 rows-per-tree / 8192 samples measured better separation
# (higher ROC-AUC) than 200/4096 in the earlier retrain - see retraining_hybrid/if.log.
IF_N_ESTIMATORS = 300
IF_MAX_SAMPLES = 8192
IF_CONTAMINATION = 0.02

# XGBoost (final, only, classifier - sees the 63 features + the Isolation Forest score)
XGB_PARAMS = dict(objective="multi:softprob", n_estimators=200, learning_rate=0.1, max_depth=10,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)

SCORE_COLUMN_NAME = "IF_Anomaly_Score"


def load(name, mmap=True):
    return np.load(os.path.join(DATA, f"{name}.npy"), mmap_mode="r" if mmap else None)


def meta():
    return json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))


def log_features(X, keep):
    """keep = indices of the non-constant columns (found in step 2 on the Benign training rows)."""
    return np.log1p(np.asarray(X[:, keep], dtype=np.float32) + 1.0)


def anomaly_scores(model, scaler, X, keep, chunk=CHUNK_ROWS):
    """Isolation Forest anomaly score for every row, in chunks. Higher = more unusual."""
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), chunk):
        part = log_features(X[s:s + chunk], keep)
        out[s:s + chunk] = -model.score_samples(scaler.transform(part))
    return out
