"""Step 1 - Build the 80/20 train/test split (read in chunks, saved to data/).

Keeps every attack row + a random 1,000,000 Benign rows, splits each class
80% train / 20% test (stratified, seed 42). Both levels use this split.

Outputs (data/): X_train.npy y_train.npy X_test.npy y_test.npy meta.json
"""
from xgb_common import BENIGN_SAMPLE, CHUNK_ROWS, DATA, DATA_CSV, SEED, SUMMARY, TEST_SIZE  # keep first (WMI)

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    t0 = time.time()
    os.makedirs(DATA, exist_ok=True)
    header = pd.read_csv(DATA_CSV, nrows=0).columns.tolist()
    feats = [c for c in header if c != "Label"]
    n_benign = json.load(open(SUMMARY, encoding="utf-8"))["label_counts"]["Benign"]
    keep_prob = min(1.0, BENIGN_SAMPLE / n_benign)
    rng = np.random.default_rng(SEED)

    xs, ys, names = [], [], {}
    for i, chunk in enumerate(pd.read_csv(DATA_CSV, dtype={c: np.float32 for c in feats}, chunksize=CHUNK_ROWS)):
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
    classes = sorted(names, key=names.get)              # classes[code] = label name

    itr, ite = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
    np.save(os.path.join(DATA, "X_train.npy"), X[itr]); np.save(os.path.join(DATA, "y_train.npy"), y[itr])
    np.save(os.path.join(DATA, "X_test.npy"), X[ite]);  np.save(os.path.join(DATA, "y_test.npy"), y[ite])
    json.dump({"features": feats, "classes": classes, "benign_code": classes.index("Benign"),
               "train_rows": int(len(itr)), "test_rows": int(len(ite))},
              open(os.path.join(DATA, "meta.json"), "w"), indent=2)
    print(pd.Series(y).map(dict(enumerate(classes))).value_counts().to_string())
    print(f"\nTrain {len(itr):,} | Test {len(ite):,} | saved to {DATA}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
