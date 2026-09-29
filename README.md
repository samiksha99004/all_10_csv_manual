# CICIDS2018 Intrusion Detection: Data Cleaning and an XGBoost Attack Classifier

This project cleans the full **CSE-CIC-IDS2018** network-traffic dataset (10 daily CSV files, 16.2 million flows) into one machine-learning-ready file. It then trains an XGBoost classifier that labels each flow as Benign or one of 14 attack types (15 classes).

Each row of the dataset is one network flow (one connection) described by 78 statistics from CICFlowMeter, plus a `Label`.

## Results at a glance

| Model | Test rows | Overall accuracy | Notes |
|---|---|---|---|
| All attacks, 15 classes (`training_xgboost/`) | 465,837 | 95.7% | 10 of 15 classes catch at least 95% of their attacks |

Five classes are below 95% recall:
- **Infilteration (51.8%):** its flows look like normal traffic.
- **Brute Force -XSS (93.3%), SQL Injection (73.3%), FTP-BruteForce (25.0%), DoS-SlowHTTPTest (0%):** after duplicates were removed, each has only 39–226 rows.

Full per-class results are in [`training_xgboost/training_conclusion.pdf`](training_xgboost/training_conclusion.pdf).

## Repository layout

```
.
├── README.md, CLAUDE.md, .gitignore
├── cleaning_dataset/     10-step cleaning pipeline (+ common.py helpers),
│                         records of the run (row counts, data types, summary, validation report)
│                         and cleaning_dataset_report.pdf
├── training_xgboost/     15-class model: training, class-weight tuning, final test,
│                         the trained model, results and training_conclusion.pdf
├── xgboost_isolation_forest_training/   hybrid: Isolation Forest flags attacks, XGBoost names them;
│                                         how it works and how to run it: steps.txt
├── notebooks/            the Kaggle decision-tree notebook this project started from
└── original_csv/         the 10 raw CSVs + the cleaned CSV (local only, not on GitHub)
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

## Training the model

```bash
# train (~17 min), tune class weights (~3 min), final test (~2 min)
python training_xgboost/train_all_attacks.py
python training_xgboost/tune_class_weights.py
python training_xgboost/evaluate_final.py
```

The model trains on a stratified 80% of each class and is tested on the remaining 20%. Benign is sampled to 1,000,000 rows, because the full 10.5 million rows don't fit in memory.

## Hybrid: Isolation Forest + XGBoost

`xgboost_isolation_forest_training/` chains two models:
1. **Isolation Forest**, trained only on normal traffic, flags anything unusual.
2. **XGBoost** names the attack for every flagged flow, or answers "Benign" to cancel a false alarm.

Both use the same 80/20 split per class. Run the five scripts in order; `steps.txt` in that folder explains the training and the run order:

```bash
python xgboost_isolation_forest_training/1_prepare_data.py            # 80/20 split, read in chunks
python xgboost_isolation_forest_training/2_train_isolation_forest.py  # stage 1 + thresholds
python xgboost_isolation_forest_training/3_evaluate_isolation_forest.py
python xgboost_isolation_forest_training/4_train_xgboost.py           # stage 2
python xgboost_isolation_forest_training/5_evaluate_hybrid.py         # final result: hybrid_report.txt
```

## Using the 15-class model

```python
import json
import numpy as np
import pandas as pd
import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model("training_xgboost/all_attacks_xgb.json")
features = json.load(open("training_xgboost/model_features.json"))["features"]   # 70 columns, in order
classes = json.load(open("training_xgboost/label_classes.json"))                 # "0" -> "Benign", ...
weights = json.load(open("training_xgboost/class_weights.json"))                 # tuned per-class weights
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
