"""
Evaluate the binary hybrid model (Isolation Forest score + XGBoost) on test_dataset.csv
ONLY - the held-out split the model never saw during training. Unlike evaluating on
merged_final.csv (which includes the training rows and gives inflated, leaked numbers),
these are the honest generalization numbers to put in your report.

Streams in chunks so memory stays flat. Produces overall accuracy/precision/recall/F1/
confusion matrix, and a per-attack-type detected-vs-missed breakdown.
"""

import pandas as pd
import numpy as np
import joblib
import os
import time

INPUT_DIR = r"F:\all_10_csv_manual"
DATA_FILE = os.path.join(INPUT_DIR, "test_dataset.csv")  # held-out only, no leakage

FEATURES_FILE = os.path.join(INPUT_DIR, "feature_columns.pkl")
HYBRID_FEATURES_FILE = os.path.join(INPUT_DIR, "hybrid_feature_columns.pkl")
IF_MODEL_FILE = os.path.join(INPUT_DIR, "isolation_forest_model.pkl")
XGB_MODEL_FILE = os.path.join(INPUT_DIR, "xgboost_hybrid_model.pkl")
THRESHOLD_FILE = os.path.join(INPUT_DIR, "xgboost_threshold.pkl")

REPORT_TXT = os.path.join(INPUT_DIR, "test_only_evaluation_report.txt")
REPORT_CSV = os.path.join(INPUT_DIR, "per_attack_type_detection_test_only.csv")

CHUNK_SIZE = 200_000

feature_cols = joblib.load(FEATURES_FILE)
use_if_score = os.path.exists(HYBRID_FEATURES_FILE)
hybrid_cols = joblib.load(HYBRID_FEATURES_FILE) if use_if_score else feature_cols
if_model = joblib.load(IF_MODEL_FILE) if use_if_score else None
model = joblib.load(XGB_MODEL_FILE)
threshold = joblib.load(THRESHOLD_FILE)
print(f"Loaded model, IF model, threshold={threshold}, feature count={len(hybrid_cols)}")

usecols = list(dict.fromkeys(feature_cols + ["Label"]))  # dedup, keep order

# Infiltration excluded from this project
EXCLUDED_LABELS = ["Infilteration"]
excluded_rows_skipped = 0

# running totals for overall confusion matrix
tn = fp = fn = tp = 0
# per-attack-type running counts: {label: [total, detected]}
per_attack = {}

print("Streaming test_dataset.csv (held-out only) in chunks...")
start = time.time()
rows_done = 0
for chunk in pd.read_csv(DATA_FILE, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    for col in chunk.columns:
        if col == "Label":
            continue
        if chunk[col].dtype == "float64":
            chunk[col] = chunk[col].astype("float32")
        elif chunk[col].dtype == "int64":
            chunk[col] = chunk[col].astype("int32")

    for lbl in EXCLUDED_LABELS:
        keep = chunk["Label"] != lbl
        excluded_rows_skipped += (~keep).sum()
        chunk = chunk[keep]
    if len(chunk) == 0:
        continue

    raw_label = chunk["Label"]
    y_true_chunk = (raw_label != "Benign").astype("int8").to_numpy()

    X_chunk = chunk[feature_cols].replace([np.inf, -np.inf], np.nan)

    if use_if_score:
        scores = if_model.decision_function(X_chunk).astype("float32")
        X_chunk = X_chunk.copy()
        X_chunk["if_anomaly_score"] = scores
        X_chunk = X_chunk[hybrid_cols]

    y_proba = model.predict_proba(X_chunk)[:, 1]
    y_pred_chunk = (y_proba >= threshold).astype("int8")

    tp += int(((y_true_chunk == 1) & (y_pred_chunk == 1)).sum())
    tn += int(((y_true_chunk == 0) & (y_pred_chunk == 0)).sum())
    fp += int(((y_true_chunk == 0) & (y_pred_chunk == 1)).sum())
    fn += int(((y_true_chunk == 1) & (y_pred_chunk == 0)).sum())

    attack_chunk = pd.DataFrame({"label": raw_label.to_numpy(), "pred": y_pred_chunk})
    attack_chunk = attack_chunk[attack_chunk["label"] != "Benign"]
    for label, grp in attack_chunk.groupby("label"):
        total = len(grp)
        detected = int(grp["pred"].sum())
        if label not in per_attack:
            per_attack[label] = [0, 0]
        per_attack[label][0] += total
        per_attack[label][1] += detected

    rows_done += len(chunk)
    print(f"  processed {rows_done:,} rows...", end="\r")

print(f"\nDone streaming in {time.time() - start:.1f}s. Total rows used: {rows_done:,} "
      f"(excluded {excluded_rows_skipped:,} rows: {EXCLUDED_LABELS})")

# ---------- overall metrics (computed manually from tp/tn/fp/fn) ----------
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision_attack = tp / (tp + fp) if (tp + fp) else 0.0
recall_attack = tp / (tp + fn) if (tp + fn) else 0.0
f1_attack = 2 * precision_attack * recall_attack / (precision_attack + recall_attack) if (precision_attack + recall_attack) else 0.0

precision_benign = tn / (tn + fn) if (tn + fn) else 0.0
recall_benign = tn / (tn + fp) if (tn + fp) else 0.0
f1_benign = 2 * precision_benign * recall_benign / (precision_benign + recall_benign) if (precision_benign + recall_benign) else 0.0

n_benign = tn + fp
n_attack = tp + fn
macro_f1 = (f1_benign + f1_attack) / 2
weighted_f1 = (f1_benign * n_benign + f1_attack * n_attack) / (n_benign + n_attack)

# ---------- per-attack-type table ----------
rows = []
for label, (total, detected) in per_attack.items():
    missed = total - detected
    rows.append({
        "label": label,
        "total": total,
        "detected": detected,
        "missed": missed,
        "detection_rate_%": round(detected / total * 100, 2) if total else 0.0,
        "missed_rate_%": round(missed / total * 100, 2) if total else 0.0,
    })
per_attack_df = pd.DataFrame(rows).sort_values("detection_rate_%")

# ---------- print + save ----------
lines = []
lines.append(f"HELD-OUT TEST-SET EVALUATION (no leakage) - {DATA_FILE}")
lines.append(f"Total rows: {rows_done}   Benign: {n_benign}   Attack: {n_attack}")
lines.append(f"Threshold used: {threshold}\n")
lines.append(f"Accuracy: {accuracy:.4f}")
lines.append(f"Benign  -> Precision: {precision_benign:.4f}  Recall: {recall_benign:.4f}  F1: {f1_benign:.4f}")
lines.append(f"Attack  -> Precision: {precision_attack:.4f}  Recall: {recall_attack:.4f}  F1: {f1_attack:.4f}")
lines.append(f"Macro F1: {macro_f1:.4f}   Weighted F1: {weighted_f1:.4f}\n")
lines.append("Confusion Matrix (rows=actual, cols=predicted) [Benign, Attack]:")
lines.append(f"[[{tn}, {fp}],")
lines.append(f" [{fn}, {tp}]]")
lines.append("\nPer-attack-type detection (sorted worst -> best):")
lines.append(per_attack_df.to_string(index=False))

report_str = "\n".join(lines)
print("\n" + report_str)

with open(REPORT_TXT, "w") as f:
    f.write(report_str)
per_attack_df.to_csv(REPORT_CSV, index=False)

print(f"\nText report saved to: {REPORT_TXT}")
print(f"Per-attack-type CSV saved to: {REPORT_CSV}")