"""Step 3 - attach the Isolation Forest score as one extra column.

Builds the actual input XGBoost trains and tests on: the 63 original features (in their
original order) plus the Isolation Forest score as column 64. This is what makes it a
STACKED model - XGBoost sees the anomaly score as one more number to split on, for every
row, not as a gate that decides which rows it gets to see.

Outputs: data/X_train_stacked.npy, data/X_test_stacked.npy, data/meta_stacked.json
"""
from h2_common import DATA, SCORE_COLUMN_NAME, load, meta  # keep first (WMI)

import json
import os

import numpy as np


def main():
    m = meta()
    X_train = load("X_train", mmap=False)
    X_test = load("X_test", mmap=False)
    train_scores = load("train_scores", mmap=False)
    test_scores = load("test_scores", mmap=False)

    X_train_stacked = np.hstack([X_train, train_scores.reshape(-1, 1)]).astype(np.float32)
    X_test_stacked = np.hstack([X_test, test_scores.reshape(-1, 1)]).astype(np.float32)
    np.save(os.path.join(DATA, "X_train_stacked.npy"), X_train_stacked)
    np.save(os.path.join(DATA, "X_test_stacked.npy"), X_test_stacked)

    stacked_features = m["features"] + [SCORE_COLUMN_NAME]
    json.dump({"features": stacked_features, "classes": m["classes"], "benign_code": m["benign_code"],
               "train_rows": m["train_rows"], "test_rows": m["test_rows"],
               "note": f"column {len(stacked_features) - 1} ('{SCORE_COLUMN_NAME}') is the Isolation Forest score"},
              open(os.path.join(DATA, "meta_stacked.json"), "w", encoding="utf-8"), indent=2)
    print(f"X_train_stacked: {X_train_stacked.shape} | X_test_stacked: {X_test_stacked.shape}")
    print(f"column {X_train_stacked.shape[1] - 1} is '{SCORE_COLUMN_NAME}'")


if __name__ == "__main__":
    main()
