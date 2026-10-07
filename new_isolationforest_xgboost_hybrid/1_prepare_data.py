"""Step 1 - sample and 80/20 split, from cleaned_dataset_2.csv (64 columns: 63 features + Label).

Keeps every attack row plus 1,000,000 random Benign rows, splits 80/20 stratified (seed 42).
Labels are kept exactly as they are in the cleaned dataset - NO merging (SSH-Bruteforce,
FTP-BruteForce, Brute Force -Web, Brute Force -XSS, the DoS types and the DDoS types all
stay as separate classes), so this gives the original 15 classes.

Outputs: data/X_train.npy, y_train.npy, X_test.npy, y_test.npy, meta.json
"""
from h2_common import BENIGN_SAMPLE, CHUNK_ROWS, CLEAN_CSV, DATA, SEED, SUMMARY, TEST_SIZE  # keep first (WMI)

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    t0 = time.time()
    os.makedirs(DATA, exist_ok=True)
    header = pd.read_csv(CLEAN_CSV, nrows=0).columns.tolist()
    feats = [c for c in header if c != "Label"]
    print(f"features: {len(feats)} (no merging - labels kept as in the cleaned dataset)", flush=True)
    n_benign = json.load(open(SUMMARY, encoding="utf-8"))["label_counts"]["Benign"]
    keep_prob = min(1.0, BENIGN_SAMPLE / n_benign)
    rng = np.random.default_rng(SEED)

    xs, ys, names = [], [], {}
    for i, chunk in enumerate(pd.read_csv(CLEAN_CSV, dtype={c: np.float32 for c in feats}, chunksize=CHUNK_ROWS)):
        benign = (chunk["Label"] == "Benign").to_numpy()
        keep = ~benign | (rng.random(len(chunk)) < keep_prob)
        part = chunk[keep]
        for lab in part["Label"].unique():
            names.setdefault(lab, len(names))
        xs.append(part[feats].to_numpy(np.float32))
        ys.append(part["Label"].map(names).to_numpy(np.int8))
        if (i + 1) % 10 == 0:
            print(f"  read {(i + 1) * CHUNK_ROWS:,} rows ({time.time() - t0:.0f}s)", flush=True)
    X = np.concatenate(xs); y = np.concatenate(ys); del xs, ys
    classes = sorted(names, key=names.get)

    itr, ite = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
    np.save(os.path.join(DATA, "X_train.npy"), X[itr]); np.save(os.path.join(DATA, "y_train.npy"), y[itr])
    np.save(os.path.join(DATA, "X_test.npy"), X[ite]); np.save(os.path.join(DATA, "y_test.npy"), y[ite])
    json.dump({"features": feats, "classes": classes, "benign_code": classes.index("Benign"),
               "train_rows": int(len(itr)), "test_rows": int(len(ite))},
              open(os.path.join(DATA, "meta.json"), "w", encoding="utf-8"), indent=2)
    print(pd.Series(y).map(dict(enumerate(classes))).value_counts().to_string())
    print(f"\n{len(classes)} classes | Train {len(itr):,} | Test {len(ite):,} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
