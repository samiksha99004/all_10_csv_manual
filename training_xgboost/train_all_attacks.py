"""Multi-class attack classifier for CICIDS2018.

Classifies each network flow as Benign or one of the 14 attack types (15 classes
total). Trained on 80% of every class and tested on the remaining 20%
(stratified, so each class keeps its 80/20 split).

Data note: all rows of the 14 attack classes are used. Benign has 10.5M rows,
too many to train in this machine's memory, so a random BENIGN_SAMPLE of them is
used. Class imbalance is handled with balanced sample weights, so the rare
attack classes are not drowned out by Benign.

Model: XGBoost, objective multi:softprob, 15 classes. Features: the 78 columns
minus the 8 all-zero columns = 70.

Outputs (this folder):
  all_attacks_xgb.json        trained model
  label_classes.json          class index -> attack name
  model_features.json         the 70 feature columns, in order
  evaluation_report.txt       overall accuracy + per-class precision/recall/F1
  confusion_matrix.png        15x15 confusion matrix (row-normalised)
  per_class_recall.png        recall bar per class with the 95% line
"""
import platform
platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())  # Windows WMI hangs; disable before pandas/xgboost

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "original_csv", "cicids2018_full_cleaned.csv")

BENIGN_SAMPLE = 1_000_000     # Benign rows to keep (all attack rows are always kept)
TEST_SIZE = 0.20              # 20% of each class held out for testing
VALID_SIZE = 0.10             # 10% of the training part for early stopping
SEED = 42
READ_CHUNK_ROWS = 200_000
BENIGN = "Benign"


def load_data():
    """Read the cleaned CSV in chunks; keep all attack rows and a random Benign sample."""
    header = pd.read_csv(DATA_CSV, nrows=0).columns.tolist()
    features = [c for c in header if c != "Label"]
    dtypes = {c: np.float32 for c in features}

    # Benign count for the sampling probability (from the cleaning summary if present)
    summ = os.path.join(ROOT, "cleaning_dataset", "9_final_summary.json")
    if os.path.exists(summ):
        n_benign = json.load(open(summ, encoding="utf-8"))["label_counts"][BENIGN]
    else:
        n_benign = sum(int((c["Label"] == BENIGN).sum())
                       for c in pd.read_csv(DATA_CSV, usecols=["Label"], chunksize=1_000_000))
    keep_prob = min(1.0, BENIGN_SAMPLE / n_benign)
    rng = np.random.default_rng(SEED)

    parts = []
    for chunk in pd.read_csv(DATA_CSV, dtype=dtypes, chunksize=READ_CHUNK_ROWS):
        is_benign = (chunk["Label"] == BENIGN).to_numpy()
        keep = ~is_benign | (rng.random(len(chunk)) < keep_prob)
        if keep.any():
            parts.append(chunk[keep])
    df = pd.concat(parts, ignore_index=True)
    return df[features], df["Label"]


def save_confusion(cm, classes, path):
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(n), classes, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (row-normalised)")
    for i in range(n):
        for j in range(n):
            v = cmn[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="fraction of actual class")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def save_recall_bar(recalls, classes, path):
    order = np.argsort(recalls)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#1f9d55" if r >= 0.95 else "#c1443a" for r in recalls[order]]
    ax.barh([classes[i] for i in order], recalls[order] * 100, color=colors)
    ax.axvline(95, color="#333", linestyle="--", linewidth=1)
    ax.text(95, -0.7, "95%", fontsize=8)
    ax.set_xlabel("Recall (%)"); ax.set_xlim(0, 100)
    ax.set_title("Per-class recall  (green = meets 95%)")
    for i, idx in enumerate(order):
        ax.text(recalls[idx]*100 + 0.5, i, f"{recalls[idx]*100:.1f}", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def prepare_splits():
    """Load the data and return the fixed fit / validation / test split.

    Deterministic (fixed seeds), so tune_class_weights.py gets the same split.
    """
    t0 = time.time()
    print(f"Loading data from {DATA_CSV} ...", flush=True)
    X, y_text = load_data()
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    classes = list(le.classes_)
    counts = pd.Series(y_text).value_counts()
    print(f"  {len(y):,} rows, {len(classes)} classes  ({time.time()-t0:.0f}s)", flush=True)
    print(counts.to_string(), flush=True)

    # drop columns that are constant in the working set (the 8 all-zero columns)
    constant = [c for c in X.columns if X[c].nunique() <= 1]
    features = [c for c in X.columns if c not in constant]
    X = X[features]
    print(f"\nDropped {len(constant)} constant columns; using {len(features)} features", flush=True)

    # stratified 80/20, then 10% of train for early stopping
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=VALID_SIZE, stratify=y_tr, random_state=SEED)
    print(f"Split: fit {len(y_fit):,} | validation {len(y_val):,} | test {len(y_te):,}", flush=True)
    return X_fit, X_val, X_te, y_fit, y_val, y_te, classes, features, len(y)


def main():
    t0 = time.time()
    X_fit, X_val, X_te, y_fit, y_val, y_te, classes, features, n_rows = prepare_splits()

    # Milder-than-balanced weights: sqrt dampens the rare/common ratio so the rare
    # classes are lifted without collapsing Benign precision (full balancing tanked it).
    w_fit = np.sqrt(compute_sample_weight("balanced", y_fit)).astype(np.float32)

    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=len(classes),
        n_estimators=500, learning_rate=0.1, max_depth=10,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        min_child_weight=1, eval_metric="mlogloss", early_stopping_rounds=40,
        n_jobs=-1, random_state=SEED,
    )
    print("Training XGBoost multi-class ...", flush=True)
    t1 = time.time()
    # Early stopping on the UNWEIGHTED validation loss (weighted loss stopped far too early).
    model.fit(X_fit, y_fit, sample_weight=w_fit,
              eval_set=[(X_val, y_val)], verbose=25)
    print(f"  best iteration {model.best_iteration}  ({time.time()-t1:.0f}s)", flush=True)

    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    bal = balanced_accuracy_score(y_te, pred)
    rec = recall_score(y_te, pred, average=None, labels=range(len(classes)), zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=range(len(classes)))

    model.save_model(os.path.join(HERE, "all_attacks_xgb.json"))
    json.dump({str(i): c for i, c in enumerate(classes)},
              open(os.path.join(HERE, "label_classes.json"), "w"), indent=2)
    json.dump({"features": features, "n_classes": len(classes)},
              open(os.path.join(HERE, "model_features.json"), "w"), indent=2)
    save_confusion(cm, classes, os.path.join(HERE, "confusion_matrix.png"))
    save_recall_bar(rec, classes, os.path.join(HERE, "per_class_recall.png"))

    below = [(classes[i], rec[i]) for i in range(len(classes)) if rec[i] < 0.95]
    lines = [
        "CICIDS2018 multi-class attack classifier (XGBoost)",
        "=" * 64,
        f"Rows: {n_rows:,}   Classes: {len(classes)}   Features: {len(features)}",
        f"Benign sampled to {BENIGN_SAMPLE:,}; all attack rows kept. Split 80/20 per class.",
        f"Trees built: {model.best_iteration}",
        "",
        f"Overall accuracy:        {acc:.4f}",
        f"Balanced accuracy (mean recall): {bal:.4f}",
        f"Classes with recall < 95%: {len(below)}",
    ]
    for c, r in sorted(below, key=lambda t: t[1]):
        lines.append(f"    {c:<26} recall {r*100:5.1f}%   (test support {int(cm[classes.index(c)].sum()):,})")
    lines += ["", "Per-class report (test set):", ""]
    lines.append(classification_report(y_te, pred, labels=range(len(classes)),
                                        target_names=classes, digits=4, zero_division=0))
    report = "\n".join(lines)
    print("\n" + report, flush=True)
    open(os.path.join(HERE, "evaluation_report.txt"), "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved model, report and plots to {HERE}  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
