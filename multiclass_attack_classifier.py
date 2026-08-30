"""
Multi-class version: predicts Benign or the SPECIFIC attack type (DoS, DDoS, Brute Force,
Infiltration, Bot, etc. - whatever exact labels are in your CICIDS2018 Label column), not
just a binary Attack/Benign flag.

Key differences from the binary script:
  - Trains on the raw "Label" column (not Label_binary)
  - Uses LabelEncoder so it works with whatever exact label strings your CSV has, without
    you hardcoding them
  - scale_pos_weight doesn't exist for multi-class - imbalance is handled with balanced
    per-sample weights instead (rare attack types like SQL Injection get upweighted)
  - No single "threshold" to tune (that was a binary-only concept) - the model just
    predicts the highest-probability class per row

Same inf-cleaning and optional IF-anomaly-score feature as before.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
import joblib
import os
import time

INPUT_DIR = r"F:\all_10_csv_manual"
TRAIN_FILE = os.path.join(INPUT_DIR, "train_dataset.csv")
TEST_FILE = os.path.join(INPUT_DIR, "test_dataset.csv")

IF_MODEL_FILE = os.path.join(INPUT_DIR, "isolation_forest_model.pkl")
FEATURES_FILE = os.path.join(INPUT_DIR, "feature_columns.pkl")

XGB_MODEL_OUTPUT = os.path.join(INPUT_DIR, "xgboost_multiclass_model.pkl")
LABEL_ENCODER_OUTPUT = os.path.join(INPUT_DIR, "label_encoder.pkl")

RANDOM_SEED = 42
CHUNK_SIZE = 300_000
USE_IF_SCORE_AS_FEATURE = True


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
print(f"Train shape: {train_df.shape}")

print("Loading test dataset (chunked, downcasted)...")
test_df = load_csv_memory_efficient(TEST_FILE)
print(f"Test shape: {test_df.shape}")

print("\nLabel distribution in train set:")
print(train_df["Label"].value_counts())

feature_cols = joblib.load(FEATURES_FILE)
print(f"\nLoaded {len(feature_cols)} feature columns from {FEATURES_FILE}")

X_train = train_df[feature_cols].copy()
raw_y_train = train_df["Label"].copy()
del train_df

X_test = test_df[feature_cols].copy()
raw_y_test = test_df["Label"].copy()
del test_df

# clean inf without a full float64 copy of the frame
for df_, name in ((X_train, "train"), (X_test, "test")):
    inf_count = sum(np.isinf(df_[c]).sum() for c in df_.columns)
    if inf_count:
        print(f"Replacing {inf_count} inf values with NaN in {name} set before training.")
    df_.replace([np.inf, -np.inf], np.nan, inplace=True)

if USE_IF_SCORE_AS_FEATURE:
    print("\nLoading Isolation Forest model to generate anomaly-score feature...")
    if_model = joblib.load(IF_MODEL_FILE)

    def if_score_in_batches(if_model, X, batch_size=200_000):
        n = len(X)
        scores = np.empty(n, dtype="float32")
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            scores[start:end] = if_model.decision_function(X.iloc[start:end]).astype("float32")
        return scores

    X_train["if_anomaly_score"] = if_score_in_batches(if_model, X_train[feature_cols])
    X_test["if_anomaly_score"] = if_score_in_batches(if_model, X_test[feature_cols])
    del if_model
    print("Added if_anomaly_score as an extra feature.")

# encode string labels -> integers, fit on train only
le = LabelEncoder()
y_train = le.fit_transform(raw_y_train)
class_names = list(le.classes_)
print(f"\nClasses ({len(class_names)}): {class_names}")

# any test-set label the encoder never saw in train gets dropped (can't score
# a class the model was never trained on) - report if that happens
known_mask = raw_y_test.isin(class_names)
if (~known_mask).sum() > 0:
    print(f"WARNING: dropping {(~known_mask).sum()} test rows with labels unseen in train: "
          f"{sorted(set(raw_y_test[~known_mask]))}")
X_test = X_test[known_mask]
y_test = le.transform(raw_y_test[known_mask])

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(class_names),
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

print("\nTraining XGBoost (multi-class)...")
start = time.time()
model.fit(X_train, y_train, sample_weight=sample_weights)
print(f"Training completed in {time.time() - start:.2f} seconds")

print("\nEvaluating on test dataset...")
y_pred = model.predict(X_test)

print(f"\nMacro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")

print("\nFull Classification Report (per attack type):")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0))

print("Confusion Matrix (rows=actual, cols=predicted):")
print(f"Order: {class_names}")
print(confusion_matrix(y_test, y_pred))

print("\nTop 15 features by importance:")
importances = model.feature_importances_
feat_names = list(X_train.columns)
top_idx = np.argsort(importances)[::-1][:15]
for i in top_idx:
    print(f"  {feat_names[i]:<30} {importances[i]:.4f}")

joblib.dump(model, XGB_MODEL_OUTPUT)
joblib.dump(le, LABEL_ENCODER_OUTPUT)
print(f"\nModel saved to: {XGB_MODEL_OUTPUT}")
print(f"Label encoder saved to: {LABEL_ENCODER_OUTPUT}")
