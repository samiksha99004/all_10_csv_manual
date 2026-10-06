# CICIDS2018 Intrusion Detection: Cleaning and Three XGBoost / Isolation Forest Models

This project cleans the full **CSE-CIC-IDS2018** network-traffic dataset (10 daily CSV files, 16.2 million flows) into one machine-learning-ready file, then trains **three intrusion-detection models** that label each flow as Benign or one of 14 attack types (15 classes).

Each row of the dataset is one network flow (one connection) described by 78 statistics from CICFlowMeter, plus a `Label`.

## The three models

| # | Model | Folder | Overall accuracy | Classes ≥95% recall |
|---|---|---|---|---|
| 1 | **Single XGBoost** (15-class) | `1_xgboost_model/` | 95.7% | **10 / 15** |
| 2 | **Two-Level XGBoost** (attack-or-not → which attack) | `2_xgboost_model/` | **96.3%** | 9 / 15 |
| 3 | **Isolation Forest + XGBoost** (anomaly gate → which attack) | `isolation_forest_and_xgboost_model/` | 89.1% | 4 / 15 |

Tested on the same 465,837-flow held-out set. A side-by-side comparison is in [`model_comparison_report.pdf`](model_comparison_report.pdf); each folder also has its own detailed training-report PDF.

**Which to use:** Model 2 for the best overall accuracy, Model 1 for the widest class coverage and simplest deployment. Model 3 trails because its unsupervised gate drops attacks, but it is the only one that can flag unknown attack types.

**Shared limit:** all three fall below 95% on Infilteration (its flows look like normal traffic) and on SQL Injection, FTP-BruteForce and DoS-SlowHTTPTest, which have only 8–15 test rows after duplicate removal.

## Repository layout

```
.
├── README.md, CLAUDE.md, .gitignore
├── model_comparison_report.pdf          the three models compared side by side
├── cleaning_dataset/                    10-step cleaning pipeline + records + cleaning_dataset_report.pdf
├── 1_xgboost_model/                     single 15-class XGBoost + training_conclusion.pdf
├── 2_xgboost_model/                     two-level XGBoost (binary gate + attack namer) + training_report.pdf
├── isolation_forest_and_xgboost_model/  Isolation Forest gate + XGBoost + hybrid_training_conclusion.pdf, steps.txt
├── notebooks/                           the Kaggle decision-tree notebook this project started from
└── original_csv/                        the 10 raw CSVs + the cleaned CSV (local only, not on GitHub)
```

## Data

The CSV files are **not in this repository**. They are 0.1–4.8 GB each, over GitHub's 100 MB file limit, and `.gitignore` excludes them.

1. Download the 10 "Processed Traffic Data for ML Algorithms" CSV files from the [CSE-CIC-IDS2018 page](https://www.unb.ca/cic/datasets/ids-2018.html).
2. Put them in `original_csv/`.

The cleaning pipeline writes `original_csv/cicids2018_full_cleaned.csv` next to them.

## Cleaning pipeline

Run from the repository root, in order. It takes about 45 minutes and streams the data in chunks, so it fits in 12 GB of RAM.

```bash
mkdir -p intermediate
for s in 1_load_merge 2_drop_timestamp 3_remove_inf_nan_null 4_remove_corrupt_rows \
         5_remove_duplicates 6_remove_conflicting_labels 7_fix_dtypes 8_shuffle_rows \
         9_save_final_csv 10_validate_final_dataset; do
  python -u cleaning_dataset/$s.py >> intermediate/pipeline_run.log 2>&1 || break
done
```

| Step | What it does | Rows removed |
|---|---|---|
| 1 | Merge the 10 files; drop repeated header rows | 59 |
| 2 | Drop `Timestamp` (80 → 79 columns) | 0 |
| 3 | Drop rows with inf, NaN, null or empty values | 95,760 |
| 4 | Drop corrupt rows (negative values, timings over 120 s) | 15 |
| 5 | Drop exact duplicate rows | 4,221,045 |
| 6 | Drop rows whose features appear with more than one label | 43,541 |
| 7–9 | Fix data types, shuffle, save the CSV | 0 |
| 10 | Independent validation (26 checks passed, 0 failed) | 0 |

Result: **11,872,582 rows × 79 columns**, with no missing, infinite, duplicate or conflicting rows.

Every model uses a stratified 80% train / 20% test split, with Benign sampled to 1,000,000 rows (the full 10.5 million don't fit in memory) and every attack row kept.

## Training the models

**Model 1 — Single XGBoost** (train ~17 min, tune ~3 min, test ~2 min):

```bash
python 1_xgboost_model/train_all_attacks.py
python 1_xgboost_model/tune_class_weights.py
python 1_xgboost_model/evaluate_final.py
```

**Model 2 — Two-Level XGBoost** (level 1 = attack vs not, level 2 = which attack):

```bash
python 2_xgboost_model/1_prepare_data.py
python 2_xgboost_model/2_train_level1.py
python 2_xgboost_model/3_train_level2.py
python 2_xgboost_model/4_evaluate.py        # result: final_report.txt
```

**Model 3 — Isolation Forest + XGBoost** (`steps.txt` in that folder has the same list):

```bash
python isolation_forest_and_xgboost_model/1_prepare_data.py
python isolation_forest_and_xgboost_model/2_train_isolation_forest.py
python isolation_forest_and_xgboost_model/3_evaluate_isolation_forest.py
python isolation_forest_and_xgboost_model/4_train_xgboost.py
python isolation_forest_and_xgboost_model/5_evaluate_hybrid.py   # result: hybrid_report.txt
```

## Using Model 1 (single XGBoost)

```python
import json
import numpy as np
import pandas as pd
import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model("1_xgboost_model/all_attacks_xgb.json")
features = json.load(open("1_xgboost_model/model_features.json"))["features"]   # 70 columns, in order
classes = json.load(open("1_xgboost_model/label_classes.json"))                # "0" -> "Benign", ...
weights = json.load(open("1_xgboost_model/class_weights.json"))                 # tuned per-class weights
w = np.array([weights[classes[str(i)]] for i in range(len(classes))])

flows = pd.read_csv("flows.csv")                      # CICFlowMeter columns, same names as the dataset
proba = model.predict_proba(flows[features].astype("float32"))
labels = [classes[str(i)] for i in np.argmax(proba * w, axis=1)]
```

Always apply the class weights. With plain `model.predict()` you get the untuned results, where 8 of 15 classes reach 95% instead of 10.

## Requirements

- Python 3.13
- pandas 3.0, numpy 2.5, pyarrow 25, scikit-learn 1.9, xgboost 3.4.1, matplotlib, reportlab (for the PDFs)
- About 12 GB RAM and 25 GB free disk space. The cleaning steps write temporary Parquet files to `intermediate/`.

On the Windows machine this was built on, the WMI service hangs, and that freezes `import pandas` and `pip`. Every script therefore disables Python's WMI query first. See `CLAUDE.md` for details.
