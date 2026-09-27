"""XGBoost binary classifier: normal traffic (Benign) vs SSH brute force.

Uses only the Benign and SSH-Bruteforce rows of
original_csv/cicids2018_full_cleaned.csv; every other attack is ignored.

- All SSH-Bruteforce rows are used. Benign has ~10.5M rows, too many for this
  machine's memory, so a random sample of BENIGN_SAMPLE rows is used.
- Features: all 78 columns except Label, minus columns that are constant in the
  training data (the 8 all-zero columns).
- Split: stratified 80% train / 20% test; 10% of train is held out for early
  stopping. The test set is only used once, for the final evaluation.
- scale_pos_weight balances Benign vs SSH during training.

Outputs (in this folder):
  ssh_bruteforce_xgb.json    trained model (load with xgb.XGBClassifier().load_model)
  model_features.json        feature columns in the order the model expects
  evaluation_report.txt      metrics on the test set
  confusion_matrix.png
  feature_importance.png     top 20 features by gain
"""
import platform


def _wmi_disabled(*args, **kwargs):
    raise OSError("WMI query disabled")


# The Windows WMI service hangs on this machine and freezes imports of
# pandas/sklearn/xgboost; this must run before those imports.
platform._wmi_query = _wmi_disabled

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "original_csv", "cicids2018_full_cleaned.csv")
SUMMARY_JSON = os.path.join(ROOT, "intermediate", "9_final_summary.json")

NORMAL, ATTACK = "Benign", "SSH-Bruteforce"
BENIGN_SAMPLE = 1_000_000      # Benign rows to use (all SSH rows are always used)
TEST_SIZE = 0.20
VALID_SIZE = 0.10              # share of the training part used for early stopping
SEED = 42
READ_CHUNK_ROWS = 200_000


def total_benign_rows():
    """Benign row count, from the cleaning summary (or by counting if it's missing)."""
    if os.path.exists(SUMMARY_JSON):
        with open(SUMMARY_JSON, encoding="utf-8") as f:
            return json.load(f)["label_counts"][NORMAL]
    n = 0
    for chunk in pd.read_csv(DATA_CSV, usecols=["Label"], chunksize=1_000_000):
        n += int((chunk["Label"] == NORMAL).sum())
    return n


def load_data():
    """Read the cleaned CSV in chunks, keeping SSH rows and a random Benign sample."""
    header = pd.read_csv(DATA_CSV, nrows=0).columns.tolist()
    features = [c for c in header if c != "Label"]
    dtypes = {c: np.float32 for c in features}       # XGBoost works in float32 anyway
    keep_prob = min(1.0, BENIGN_SAMPLE / total_benign_rows())
    rng = np.random.default_rng(SEED)

    parts = []
    for chunk in pd.read_csv(DATA_CSV, dtype=dtypes, chunksize=READ_CHUNK_ROWS):
        is_attack = (chunk["Label"] == ATTACK).to_numpy()
        is_normal = (chunk["Label"] == NORMAL).to_numpy()
        keep = is_attack | (is_normal & (rng.random(len(chunk)) < keep_prob))
        if keep.any():
            parts.append(chunk[keep])
    df = pd.concat(parts, ignore_index=True)
    X = df[features]
    y = (df["Label"] == ATTACK).astype(np.int8).to_numpy()
    return X, y


def save_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    names = [NORMAL, ATTACK]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_feature_importance(model, path, top=20):
    gain = model.get_booster().get_score(importance_type="gain")
    imp = pd.Series(gain).sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color="#3b6ea5")
    ax.set_xlabel("Gain")
    ax.set_title(f"Top {len(imp)} features (gain)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return imp.sort_values(ascending=False)


def main():
    t0 = time.time()
    print(f"Loading {NORMAL} + {ATTACK} rows from {DATA_CSV} ...")
    X, y = load_data()
    print(f"  {len(y):,} rows: {int((y == 0).sum()):,} {NORMAL}, {int(y.sum()):,} {ATTACK}"
          f"  ({time.time() - t0:.0f}s)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VALID_SIZE, stratify=y_train, random_state=SEED)
    del X

    constant = [c for c in X_fit.columns if X_fit[c].nunique() <= 1]
    features = [c for c in X_fit.columns if c not in constant]
    print(f"Dropped {len(constant)} constant columns: {constant}")
    print(f"Using {len(features)} features")
    print(f"Split: fit {len(y_fit):,} | validation {len(y_val):,} | test {len(y_test):,}")

    scale_pos_weight = float((y_fit == 0).sum() / max(1, (y_fit == 1).sum()))
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        # early stopping follows the LAST metric; aucpr reaches 1.0 at once, so logloss goes last
        eval_metric=["aucpr", "logloss"],
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=SEED,
    )
    print(f"Training XGBoost (scale_pos_weight={scale_pos_weight:.2f}) ...")
    t1 = time.time()
    model.fit(X_fit[features], y_fit, eval_set=[(X_val[features], y_val)], verbose=50)
    print(f"  best iteration {model.best_iteration}  ({time.time() - t1:.0f}s)")

    proba = model.predict_proba(X_test[features])[:, 1]
    pred = (proba >= 0.5).astype(np.int8)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision_ssh": precision_score(y_test, pred, zero_division=0),
        "recall_ssh": recall_score(y_test, pred, zero_division=0),
        "f1_ssh": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "false_positive_rate": fp / max(1, fp + tn),
    }

    model.save_model(os.path.join(HERE, "ssh_bruteforce_xgb.json"))
    with open(os.path.join(HERE, "model_features.json"), "w", encoding="utf-8") as f:
        json.dump({"features": features, "classes": {"0": NORMAL, "1": ATTACK},
                   "threshold": 0.5}, f, indent=2)
    save_confusion_matrix(cm, os.path.join(HERE, "confusion_matrix.png"))
    top = save_feature_importance(model, os.path.join(HERE, "feature_importance.png"))

    lines = [
        f"XGBoost - {NORMAL} (0) vs {ATTACK} (1)",
        "=" * 60,
        f"Data: {len(y_fit) + len(y_val) + len(y_test):,} rows "
        f"(all {ATTACK} rows + random {NORMAL} sample, target {BENIGN_SAMPLE:,})",
        f"Train {len(y_fit):,} | validation {len(y_val):,} | test {len(y_test):,}",
        f"Features: {len(features)}   best iteration: {model.best_iteration}",
        "",
        "Test set metrics (threshold 0.5):",
        *[f"  {k:<20} {v:.6f}" for k, v in metrics.items()],
        "",
        "Confusion matrix (rows = actual, cols = predicted):",
        f"  {'':<16}{NORMAL:>16}{ATTACK:>16}",
        f"  {NORMAL:<16}{tn:>16,}{fp:>16,}",
        f"  {ATTACK:<16}{fn:>16,}{tp:>16,}",
        "",
        classification_report(y_test, pred, target_names=[NORMAL, ATTACK], digits=6),
        "Top 10 features by gain:",
        *[f"  {k:<22} {v:,.1f}" for k, v in top.head(10).items()],
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(HERE, "evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved model, report and plots to {HERE}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
