"""Shared settings and helpers for the hybrid Isolation Forest + XGBoost steps.

Import this before pandas / sklearn / xgboost (it disables the hanging WMI query).
"""
import platform
platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())  # Windows WMI hangs imports

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(HERE, "isolation_forest.joblib")
SCALER_PATH = os.path.join(HERE, "scaler.joblib")
CONFIG_PATH = os.path.join(HERE, "model_config.json")
CHUNK_ROWS = 200_000
TARGET = 0.95

# --- Isolation Forest settings ---------------------------------------------
# Fixes for the first version (which flagged 95% of normal traffic as attack):
#   1. features are standardised (zero mean, unit variance) - the single biggest
#      lever; without it the raw column ranges dominate every split.
#   2. contamination is low (2%), so the model expects few anomalies, as the IDS
#      papers recommend (2% gave >97% benign and >95% attack for them).
#   3. the alarm threshold is set so at most 5% of NORMAL traffic is flagged
#      (BENIGN_FLOOR), so "normal accuracy" is >= 95% by construction, never
#      bought by flagging everything.
N_ESTIMATORS = 300      # a few more trees for a steadier score
MAX_SAMPLES = 4096      # larger sample per tree separates dense attack floods better than 256
MAX_FEATURES = 1.0
CONTAMINATION = 0.02
N_JOBS = -1
SEED = 42
BENIGN_FLOOR = 0.95     # keep at least 95% of normal traffic un-flagged (<= 5% false alarms)

# --- Hybrid stage 2 (XGBoost) ----------------------------------------------
XGB_PATH = os.path.join(HERE, "xgboost_stage2.json")
XGB_PARAMS = dict(objective="multi:softprob", n_estimators=500, learning_rate=0.1, max_depth=10,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=N_JOBS,
                  random_state=SEED)


def load(name, mmap=True):
    """Load an array from data/ (memory-mapped, so it is read from disk in pieces)."""
    return np.load(os.path.join(DATA, f"{name}.npy"), mmap_mode="r" if mmap else None)


def meta():
    return json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))


def log_features(X, keep):
    """Keep the useful columns and compress the huge value ranges with log1p.

    Rates run 0..1.8e9, so raw values make almost every random split useless.
    x >= -1 in this data, so log1p(x + 1) is always finite. Scaling is applied
    on top of this (see transform).
    """
    return np.log1p(np.asarray(X[:, keep], dtype=np.float32) + 1.0)


def transform(X, keep, scaler):
    """log1p, then standardise with the scaler fitted on the Benign training rows."""
    return scaler.transform(log_features(X, keep)).astype(np.float32)


def anomaly_scores(model, X, keep, scaler):
    """Anomaly score for every row, in chunks. Higher = more unusual."""
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), CHUNK_ROWS):
        out[s:s + CHUNK_ROWS] = -model.score_samples(transform(X[s:s + CHUNK_ROWS], keep, scaler))
    return out
