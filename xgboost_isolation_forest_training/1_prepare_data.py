"""Step 1 - Build the 80/20 train/test split for the Isolation Forest.

Reads the cleaned CSV in chunks (never the whole file at once), keeps every
attack row plus a random Benign sample, splits each class 80% train / 20% test
(stratified, seed 42 - the same split rule as the XGBoost model), and saves the
arrays to data/ so steps 2 and 3 never have to re-read the 4.8 GB CSV.

Outputs (data/):  X_train.npy  y_train.npy  X_test.npy  y_test.npy  meta.json
"""
import platform
platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())  # Windows WMI hangs imports

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "original_csv", "cicids2018_full_cleaned.csv")
SUMMARY = os.path.join(ROOT, "cleaning_dataset", "9_final_summary.json")
OUT = os.path.join(HERE, "data")

BENIGN_SAMPLE = 1_000_000   # Benign rows kept (all 10.5M do not fit in RAM); all attack rows are kept
CHUNK_ROWS = 200_000
TEST_SIZE = 0.20
SEED = 42


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    header = pd.read_csv(DATA_CSV, nrows=0).columns.tolist()
    feats = [c for c in header if c != "Label"]
    n_benign = json.load(open(SUMMARY, encoding="utf-8"))["label_counts"]["Benign"]
    keep_prob = min(1.0, BENIGN_SAMPLE / n_benign)
    rng = np.random.default_rng(SEED)

    xs, ys, names = [], [], {}
    for i, chunk in enumerate(pd.read_csv(DATA_CSV, dtype={c: np.float32 for c in feats},
                                          chunksize=CHUNK_ROWS)):
        benign = (chunk["Label"] == "Benign").to_numpy()
        keep = ~benign | (rng.random(len(chunk)) < keep_prob)
        part = chunk[keep]
        for lab in part["Label"].unique():
            names.setdefault(lab, len(names))
        xs.append(part[feats].to_numpy(np.float32))
        ys.append(part["Label"].map(names).to_numpy(np.int8))
        if (i + 1) % 10 == 0:
            print(f"  read {(i + 1) * CHUNK_ROWS:,} rows", flush=True)
    X = np.concatenate(xs); y = np.concatenate(ys); del xs, ys

    classes = sorted(names, key=names.get)          # classes[code] = label name
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
    np.save(os.path.join(OUT, "X_train.npy"), X[idx_tr]); np.save(os.path.join(OUT, "y_train.npy"), y[idx_tr])
    np.save(os.path.join(OUT, "X_test.npy"), X[idx_te]);  np.save(os.path.join(OUT, "y_test.npy"), y[idx_te])
    json.dump({"features": feats, "classes": classes, "benign_code": classes.index("Benign"),
               "train_rows": int(len(idx_tr)), "test_rows": int(len(idx_te))},
              open(os.path.join(OUT, "meta.json"), "w"), indent=2)

    counts = pd.Series(y).map(dict(enumerate(classes))).value_counts()
    print(counts.to_string())
    print(f"\nTrain {len(idx_tr):,} rows | Test {len(idx_te):,} rows | saved to {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
