"""Shared settings for the hybrid retrain (Isolation Forest + XGBoost), on cleaned_dataset_2.csv.
Import this first (WMI patch for Windows)."""
import platform
platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CLEAN_CSV = r"F:\all_10_csv_manual\original_csv\cleaned_dataset_2.csv"   # 64 columns: 63 features + Label
SUMMARY = r"F:\all_10_csv_manual\cleaning_dataset\9_final_summary.json"

BENIGN_SAMPLE = 1_000_000
CHUNK_ROWS = 200_000
TEST_SIZE = 0.20
SEED = 42
TARGET = 0.95

# Isolation Forest (stage 1)
N_ESTIMATORS = 300       # raised from 200 to try to improve stage-1 accuracy (v1 backed up in v1_backup/)
MAX_SAMPLES = 8192       # raised from 4096, same reason
CONTAMINATION = 0.02

# XGBoost (stage 2)
XGB_PARAMS = dict(objective="multi:softprob", n_estimators=200, learning_rate=0.1, max_depth=10,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)


def load(name, mmap=True):
    return np.load(os.path.join(DATA, f"{name}.npy"), mmap_mode="r" if mmap else None)


def meta():
    return json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))


def log_features(X, keep):
    return np.log1p(np.asarray(X[:, keep], dtype=np.float32) + 1.0)


def transform(X, keep, scaler):
    return scaler.transform(log_features(X, keep)).astype(np.float32)


def anomaly_scores(model, X, keep, scaler):
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), CHUNK_ROWS):
        out[s:s + CHUNK_ROWS] = -model.score_samples(transform(X[s:s + CHUNK_ROWS], keep, scaler))
    return out
