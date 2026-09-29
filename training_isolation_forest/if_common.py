"""Shared settings and helpers for the Isolation Forest steps (import before pandas/sklearn)."""
import platform
platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())  # Windows WMI hangs imports

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(HERE, "isolation_forest.joblib")
CONFIG_PATH = os.path.join(HERE, "model_config.json")
CHUNK_ROWS = 200_000
TARGET = 0.95

# Isolation Forest settings (see steps.txt for where they come from)
N_ESTIMATORS = 200      # 100-200 trees is the usual sweet spot; more adds stability, not accuracy
MAX_SAMPLES = 256       # rows per tree, as in the original paper (Liu et al. 2008)
MAX_FEATURES = 1.0      # each tree may split on any feature
N_JOBS = -1             # use every CPU core for fitting and scoring
SEED = 42


def load(name, mmap=True):
    """Load an array from data/ (memory-mapped, so it is read from disk in pieces)."""
    return np.load(os.path.join(DATA, f"{name}.npy"), mmap_mode="r" if mmap else None)


def meta():
    return json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))


def transform(X, keep):
    """Keep the useful columns and compress the huge value ranges.

    Isolation Forest picks split points uniformly between a feature's min and
    max. Byte and packet rates span 0 to ~1.8e9, so almost every split would land
    in empty space. log1p(x + 1) spreads the values out (x >= -1 in this data,
    so the result is always finite).
    """
    return np.log1p(np.asarray(X[:, keep], dtype=np.float32) + 1.0)


def anomaly_scores(model, X, keep):
    """Anomaly score for every row, computed in chunks. Higher = more unusual."""
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), CHUNK_ROWS):
        out[s:s + CHUNK_ROWS] = -model.score_samples(transform(X[s:s + CHUNK_ROWS], keep))
    return out


def per_class_rates(scores, y, threshold, benign):
    """Share of each class flagged as attack (score >= threshold).

    For attack classes this is recall (attacks caught); for Benign it is the
    false-alarm rate, so Benign's 'correct' rate is 1 minus it.
    """
    flagged = scores >= threshold
    rates = {}
    for c in np.unique(y):
        m = y == c
        caught = float(flagged[m].mean())
        rates[int(c)] = 1.0 - caught if c == benign else caught
    return rates, flagged
