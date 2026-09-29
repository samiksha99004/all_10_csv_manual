"""Shared settings and data prep for the two-level XGBoost (import before pandas/xgboost)."""
import platform
platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())  # Windows WMI hangs imports

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "original_csv", "cicids2018_full_cleaned.csv")
SUMMARY = os.path.join(ROOT, "cleaning_dataset", "9_final_summary.json")
DATA = os.path.join(HERE, "data")
L1_PATH = os.path.join(HERE, "level1_binary_xgb.json")     # attack vs not
L2_PATH = os.path.join(HERE, "level2_multiclass_xgb.json")  # which attack

BENIGN_SAMPLE = 1_000_000
CHUNK_ROWS = 200_000
TEST_SIZE = 0.20
SEED = 42

# same XGBoost settings as training_xgboost/ (reached 95.7% there)
PARAMS = dict(learning_rate=0.1, max_depth=10, subsample=0.8, colsample_bytree=0.8,
              tree_method="hist", n_jobs=-1, random_state=SEED)
N_ESTIMATORS = 500


def load(name, mmap=True):
    return np.load(os.path.join(DATA, f"{name}.npy"), mmap_mode="r" if mmap else None)


def meta():
    return json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
