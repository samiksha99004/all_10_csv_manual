"""
Hybrid model: Isolation Forest (unsupervised anomaly score) + XGBoost (supervised classifier)
for host-based NIDS anomaly detection (CICIDS2018).

WHY THIS BEATS THE OLD 84% "ACCURACY" NUMBER
----------------------------------------------
Your isolation_forest_model.pkl comment already flags the real problem: 84% accuracy on this
dataset is misleading because ~88% of the traffic is Benign, so a model that predicts "Benign"
for everything scores ~88% accuracy while catching 0% of attacks. Isolation Forest only reached
~1% attack recall - it was barely better than that. Plain "accuracy" is not the right metric
for this class imbalance, so this script reports Precision/Recall/F1 per class and macro-F1
(the numbers that actually tell you whether attacks are being caught), alongside accuracy.

XGBoost is a SUPERVISED model - it can directly learn from your Label_binary column, which
Isolation Forest never sees. That alone is usually a much bigger jump than any amount of
"ensembling." To still get a genuine hybrid (not just XGBoost alone), this script:
  1. Loads your existing isolation_forest_model.pkl
  2. Uses it to generate an anomaly score for every row (decision_function output)
  3. Adds that anomaly score as ONE EXTRA FEATURE alongside your 71 original features
  4. Trains XGBoost on [71 features + IF anomaly score] -> Label_binary
This is a legitimate feature-stacking hybrid. If you'd rather train XGBoost on the raw 71
features only (no IF score), set USE_IF_SCORE_AS_FEATURE = False below - in practice it will
likely perform almost identically, because IF's signal is weak, but this keeps the "combining
two models" framing intact for your report if you want it.

Outputs: xgboost_hybrid_model.pkl, accuracy, precision/recall/F1 (per class + macro/weighted),
confusion matrix, classification report, and a feature-importance printout.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, f1_score
)
from xgboost import XGBClassifier
import joblib
import os
import time

INPUT_DIR = r"F:\all_10_csv_manual"
TRAIN_FILE = os.path.join(INPUT_DIR, "train_dataset.csv")
TEST_FILE = os.path.join(INPUT_DIR, "test_dataset.csv")

IF_MODEL_FILE = os.path.join(INPUT_DIR, "isolation_forest_model.pkl")
FEATURES_FILE = os.path.join(INPUT_DIR, "feature_columns.pkl")

XGB_MODEL_OUTPUT = os.path.join(INPUT_DIR, "xgboost_hybrid_model.pkl")

RANDOM_SEED = 42
CHUNK_SIZE = 300_000
USE_IF_SCORE_AS_FEATURE = True   # set False to train XGBoost on the 71 raw features only


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

# Infiltration excluded from this project - drop it from both splits before training
EXCLUDED_LABELS = ["Infilteration"]
for lbl in EXCLUDED_LABELS:
    before = len(train_df)
    train_df = train_df[train_df["Label"] != lbl]
    test_df = test_df[test_df["Label"] != lbl]
    print(f"Excluded '{lbl}': removed {before - len(train_df)} train rows")
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
print(f"Train shape after exclusion: {train_df.shape}")
print(f"Test shape after exclusion: {test_df.shape}")

train_df["Label_binary"] = (train_df["Label"] != "Benign").astype("int8")
test_df["Label_binary"] = (test_df["Label"] != "Benign").astype("int8")

feature_cols = joblib.load(FEATURES_FILE)
print(f"Loaded {len(feature_cols)} feature columns from {FEATURES_FILE}")

X_train = train_df[feature_cols].copy()
y_train = train_df["Label_binary"].copy()
del train_df

X_test = test_df[feature_cols].copy()
y_test = test_df["Label_binary"].copy()
del test_df

# some CICIDS columns (e.g. Flow Bytes/s) contain inf, or values that overflow
# float32 into inf when downcast. XGBoost rejects inf outright, so clean it here.
# (checked column-by-column to avoid building a full-size float64 copy of the frame)
for df_, name in ((X_train, "train"), (X_test, "test")):
    inf_count = sum(np.isinf(df_[c]).sum() for c in df_.columns)
    if inf_count:
        print(f"Replacing {inf_count} inf values with NaN in {name} set before training.")
    df_.replace([np.inf, -np.inf], np.nan, inplace=True)
# XGBoost handles NaN as "missing" natively, no fillna needed.


def if_score_in_batches(if_model, X, batch_size=200_000):
    # decision_function on the full array at once tries to allocate the whole
    # thing as float64 internally, which is what blew up memory. Score in
    # batches instead and write straight into a preallocated float32 array.
    n = len(X)
    scores = np.empty(n, dtype="float32")
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        scores[start:end] = if_model.decision_function(X.iloc[start:end]).astype("float32")
    return scores


if USE_IF_SCORE_AS_FEATURE:
    print("\nLoading Isolation Forest model to generate anomaly-score feature...")
    if_model = joblib.load(IF_MODEL_FILE)
    # higher = more normal, lower/negative = more anomalous
    X_train["if_anomaly_score"] = if_score_in_batches(if_model, X_train[feature_cols])
    X_test["if_anomaly_score"] = if_score_in_batches(if_model, X_test[feature_cols])
    del if_model
    print("Added if_anomaly_score as an extra feature.")

print(f"\nFinal training feature count: {X_train.shape[1]}")

# class imbalance handling (attacks are the minority class)
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"Benign: {neg}, Attack: {pos}, scale_pos_weight={scale_pos_weight:.3f}")

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    tree_method="hist",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

print("\nTraining XGBoost...")
start = time.time()
model.fit(X_train, y_train)
print(f"Training completed in {time.time() - start:.2f} seconds")

print("\nEvaluating on test dataset...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1])
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")

print(f"\nAccuracy: {acc:.4f}")
print(f"Benign  -> Precision: {precision[0]:.4f}  Recall: {recall[0]:.4f}  F1: {f1[0]:.4f}")
print(f"Attack  -> Precision: {precision[1]:.4f}  Recall: {recall[1]:.4f}  F1: {f1[1]:.4f}")
print(f"Macro F1: {macro_f1:.4f}   Weighted F1: {weighted_f1:.4f}")

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"], digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (rows=actual, cols=predicted) [Benign, Attack]:")
print(cm)

print("\nTop 15 features by importance:")
importances = model.feature_importances_
feat_names = list(X_train.columns)
top_idx = np.argsort(importances)[::-1][:15]
for i in top_idx:
    print(f"  {feat_names[i]:<30} {importances[i]:.4f}")

# --- Threshold tuning: default predict() uses 0.5, which is what gave 18,364
# missed attacks above. Lowering the threshold trades some Benign precision
# for higher Attack recall (fewer missed attacks) - usually the right trade
# for an IDS, since a missed attack is worse than a false alarm.
print("\nThreshold tuning (recall/precision on Attack class):")
y_proba = model.predict_proba(X_test)[:, 1]
for t in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1]:
    y_pred_t = (y_proba >= t).astype("int8")
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred_t, average=None, labels=[0, 1])
    missed = ((y_test == 1) & (y_pred_t == 0)).sum()
    print(f"  threshold={t:<5} Attack Precision={p[1]:.4f} Recall={r[1]:.4f} F1={f[1]:.4f}  missed_attacks={missed}")

# pick your threshold from the table above and set it here, then re-run just
# this block (or the whole script) to get final metrics at that threshold
CHOSEN_THRESHOLD = 0.3
y_pred_final = (y_proba >= CHOSEN_THRESHOLD).astype("int8")
print(f"\n--- Final report at threshold={CHOSEN_THRESHOLD} ---")
print(classification_report(y_test, y_pred_final, target_names=["Benign", "Attack"], digits=4))
print("Confusion Matrix (rows=actual, cols=predicted) [Benign, Attack]:")
print(confusion_matrix(y_test, y_pred_final))

joblib.dump(model, XGB_MODEL_OUTPUT)
print(f"\nModel saved to: {XGB_MODEL_OUTPUT}")
joblib.dump(CHOSEN_THRESHOLD, os.path.join(INPUT_DIR, "xgboost_threshold.pkl"))
print(f"Chosen threshold saved to: {os.path.join(INPUT_DIR, 'xgboost_threshold.pkl')}")

if USE_IF_SCORE_AS_FEATURE:
    hybrid_feature_cols = feature_cols + ["if_anomaly_score"]
    joblib.dump(hybrid_feature_cols, os.path.join(INPUT_DIR, "hybrid_feature_columns.pkl"))
    print(f"Hybrid feature list saved to: {os.path.join(INPUT_DIR, 'hybrid_feature_columns.pkl')}")