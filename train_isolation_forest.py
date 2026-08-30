"""
Train an Isolation Forest model for host-based NIDS anomaly detection (CICIDS2018).

This is the ORIGINAL version that produced 84% overall accuracy but only 1% attack recall.
NOTE: This model barely detects real attacks - 84% accuracy is misleading due to class
imbalance (88% of traffic is Benign, so predicting "Benign" always already scores ~88%).
Kept here on request for reference/comparison purposes.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import time

INPUT_DIR = r"F:\all_10_csv_manual"
TRAIN_FILE = os.path.join(INPUT_DIR, "train_dataset.csv")
TEST_FILE = os.path.join(INPUT_DIR, "test_dataset.csv")

MODEL_OUTPUT = os.path.join(INPUT_DIR, "isolation_forest_model.pkl")
FEATURES_OUTPUT = os.path.join(INPUT_DIR, "feature_columns.pkl")

RANDOM_SEED = 42
CONTAMINATION = 0.05
N_ESTIMATORS = 100
CHUNK_SIZE = 300_000


def load_csv_memory_efficient(filepath):
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        for col in chunk.columns:
            if col == "Label":
                continue
            if chunk[col].dtype == "float64":
                chunk[col] = chunk[col].astype("float32")
            elif chunk[col].dtype == "int64":
                chunk[col] = chunk[col].astype("int32")
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


print("Loading train dataset (chunked, downcasted)...")
train_df = load_csv_memory_efficient(TRAIN_FILE)
print(f"Train shape: {train_df.shape}, memory usage: {train_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

print("Loading test dataset (chunked, downcasted)...")
test_df = load_csv_memory_efficient(TEST_FILE)
print(f"Test shape: {test_df.shape}, memory usage: {test_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

train_df["Label_binary"] = (train_df["Label"] != "Benign").astype("int8")
test_df["Label_binary"] = (test_df["Label"] != "Benign").astype("int8")

feature_cols = [c for c in train_df.columns if c not in ("Label", "Label_binary")]

benign_train = train_df[train_df["Label_binary"] == 0][feature_cols]
print(f"\nTraining Isolation Forest on {len(benign_train)} Benign rows...")

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1
)

start = time.time()
model.fit(benign_train)
print(f"Training completed in {time.time() - start:.2f} seconds")

del benign_train, train_df

print("\nEvaluating on test dataset...")
X_test = test_df[feature_cols]
y_test = test_df["Label_binary"]

raw_preds = model.predict(X_test)
y_pred = np.where(raw_preds == -1, 1, 0)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, MODEL_OUTPUT)
print(f"\nModel saved to: {MODEL_OUTPUT}")

joblib.dump(feature_cols, FEATURES_OUTPUT)
print(f"Feature columns saved to: {FEATURES_OUTPUT}")