# CICIDS2018 – merge & clean pipeline

This folder merges the 10 daily CICIDS2018 CSV files (CICFlowMeter network-flow features) into one cleaned, **ML-ready** dataset. The output keeps **all 79 columns**: the 80 raw columns minus `Timestamp`. It has:
- no NaN, null, empty or ±inf values;
- no corrupt rows and no duplicate rows;
- no identical features carrying different labels;
- one correct data type per column;
- rows shuffled.

**Status:** cleaning is done. No model has been trained; the user decides the next steps.

## What the user wants (keep to this)
- **Clean only; never drop columns.** No feature selection, no removal of zero or correlated columns, unless the user asks for it. An earlier version dropped columns (46 left) and the user rejected it.
- The final CSV has **79 columns**: the raw columns minus `Timestamp`. There is no `source_file` column and no `time` feature. `Label` keeps the **original text** (not encoded).
- `original_csv/` must hold exactly **11 CSVs**: the 10 raw files plus `cicids2018_full_cleaned.csv`. Every other output goes in `intermediate/`.
- One script per step, named `N_stepname.py`.
- The user confirmed these removals: all exact duplicates (even though that leaves FTP-BruteForce and SlowHTTPTest with only a few dozen rows), all rows whose features appear with more than one label, and the 15 corrupt rows. Ask before removing anything else.

## Folder layout
```
all_10_csv_manual/
├── CLAUDE.md
├── cicids2018-using-decision-trees.ipynb   reference notebook (Kaggle, uses 1 file); not part of the pipeline
├── codes_for_cleaning/
│   ├── common.py                           paths (data folders are one level up), chunked Parquet I/O, log_step, WMI workaround
│   └── 1_load_merge.py … 10_validate_final_dataset.py
├── xg_bruteforce/                          XGBoost Benign-vs-SSH-Bruteforce model (see "Models" below)
├── original_csv/                           10 raw day files + cicids2018_full_cleaned.csv
└── intermediate/                           small records only: cleaning_log.json, 7_final_dtypes.json,
                                            9_final_summary.json, 10_validation_report.txt
```
The per-step Parquet files (~14 GB) and the unused `attack_csvs/` folder were deleted on request. Steps 1–9 recreate the Parquet files.

## Raw data (`original_csv/`, ~6.9 GB, 16,233,002 data rows)

| File | Rows | Classes |
|---|---|---|
| Wednesday-14-02-2018 | 1,048,575 | Benign, FTP-BruteForce, SSH-Bruteforce |
| Thursday-15-02-2018 | 1,048,575 | Benign, DoS-GoldenEye, DoS-Slowloris |
| Friday-16-02-2018 | 1,048,575 | Benign, DoS-Hulk, DoS-SlowHTTPTest (+1 stray header row) |
| Thuesday-20-02-2018 (sic) | 7,948,748 | Benign, DDoS-LOIC-HTTP. **4 GB, 84 columns** |
| Wednesday-21-02-2018 | 1,048,575 | Benign, DDOS-HOIC, DDOS-LOIC-UDP |
| Thursday-22-02-2018 | 1,048,575 | Benign, Brute Force-Web/XSS, SQL Injection |
| Friday-23-02-2018 | 1,048,575 | Benign, Brute Force-Web/XSS, SQL Injection |
| Wednesday-28-02-2018 | 613,104 | Benign, Infilteration (+33 stray header rows) |
| Thursday-01-03-2018 | 331,125 | Benign, Infilteration (+25 stray header rows) |
| Friday-02-03-2018 | 1,048,575 | Benign, Bot |

Quirks:
- **Tuesday-20-02 has 4 extra ID columns** (`Flow ID`, `Src IP`, `Src Port`, `Dst IP`) that the other 9 files lack. They are dropped so every file has the same 80 columns.
- **Stray header rows:** 59 rows repeat the header line inside the data (their `Label` is literally "Label").
- **Infinity:** `Flow Byts/s` and `Flow Pkts/s` contain `Infinity`, which appears in 95,760 rows.
- **Scientific notation:** some whole-number timing columns are stored as rounded scientific notation in some files (e.g. `1.00E+07` in `Flow IAT Max` in Friday-02-03). Their values are still whole numbers, so the data type is decided from the values, not the text. The lost precision can't be recovered.
- **Spelling:** the label is spelled "Infilteration" in the raw data; keep it.

## Environment (important)
- Python 3.13 (`C:\Users\Lenovo\AppData\Local\Programs\Python\Python313`), with pandas 3.0, numpy 2.5, pyarrow 25, scikit-learn 1.9, matplotlib and seaborn.
- **The Windows WMI service hangs on this machine.** Python's `platform` module queries WMI, so `import pandas`, `import sklearn` and `pip` freeze. PowerShell `Get-CimInstance`, `tasklist` and `wmic` hang too.
  - Every script must `from common import …` **before** importing pandas or sklearn. `common.py` makes `platform._wmi_query` raise `OSError`.
  - For ad-hoc snippets run from the project root, first do `sys.path.insert(0, "codes_for_cleaning")`, then `import common`.
  - To install packages, use this wrapper, because plain `pip install` hangs forever:
    ```python
    import platform, sys
    platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())
    from pip._internal.cli.main import main; sys.exit(main())
    ```
- **Memory is tight.** RAM is 12 GB, and Windows' commit limit (RAM plus page file) is 19.1 GB. Chrome, VS Code, Defender and similar apps often commit about 16 GB, so a step may get only about 3 GB.
  - "Unable to allocate 781 KiB" or a failure with no traceback means that limit was hit.
  - Never load the full dataset at once.
  - Stream it with `common.iter_chunks()` (100k rows) and keep accumulators small. Store 16M labels as int8 codes, not Python strings, and avoid full-chunk copies like `DataFrame.replace`.
- **pyarrow's `ParquetFile.iter_batches` leaks memory.** One pass over a 12M-row file grew to 2.8 GB. `iter_chunks` therefore reads one row group at a time with `use_threads=False`, which stays flat at about 650 MB. Don't switch it back. `common.py` also sets pyarrow to use the system allocator.
- pandas 3: text columns use the `str` dtype, not `object`. Test with `pd.api.types.is_numeric_dtype`.
- Create `intermediate/` before redirecting a log into it; the shell opens the log before Python creates the folder.

## Pipeline
Each step streams the previous step's Parquet file from `intermediate/` and writes a new one. It also records rows in and out in `intermediate/cleaning_log.json` through `log_step()`; re-running a step overwrites its entry. Numeric columns stay float64 in the intermediate files, and step 6 sets the final types.

| # | Script | Does | Output |
|---|---|---|---|
| 1 | `1_load_merge.py` | Loads the 10 CSVs, drops Tuesday's 4 ID columns and the 59 stray header rows, converts text to numbers. Gives 80 columns | `1_merged.parquet` |
| 2 | `2_drop_timestamp.py` | Drops `Timestamp`, giving 79 columns | `2_no_timestamp.parquet` |
| 3 | `3_remove_inf_nan_null.py` | Drops rows with inf, NaN, null or empty values, including an empty `Label` | `3_no_inf_nan_null.parquet` |
| 4 | `4_remove_corrupt_rows.py` | Drops corrupt rows (rule below the table) | `4_no_corrupt_rows.parquet` |
| 5 | `5_remove_duplicates.py` | Drops exact duplicates across all 79 columns, keeping the first copy | `5_no_duplicates.parquet` |
| 6 | `6_remove_conflicting_labels.py` | Drops every row whose 78 feature values also appear with a different `Label` (all copies) | `6_no_conflicting_labels.parquet` |
| 7 | `7_fix_dtypes.py` | Sets int64 if every value is whole, otherwise float64; `Label` stays text | `7_fixed_dtypes.parquet`, `7_final_dtypes.json` |
| 8 | `8_shuffle_rows.py` | Uniform random shuffle (seed 42) using about 120 random bucket files on disk | `8_shuffled.parquet` |
| 9 | `9_save_final_csv.py` | Writes the CSV and prints shape, class counts, type/min/max per column, and rows dropped per step | `original_csv/cicids2018_full_cleaned.csv`, `9_final_summary.json` |
| 10 | `10_validate_final_dataset.py` | Independent check that re-reads the CSV as raw text, including ML-readiness checks (float32 range, column names, shuffle). Reports PASS/FAIL/WARN/INFO and exits 1 on any FAIL | `10_validation_report.txt` |

Step 4's rule: a row is corrupt if it has a negative value in any column, or a timing column above the 120 s flow timeout. `Init Fwd Win Byts` and `Init Bwd Win Byts` may be −1, which means "not set".

To run everything from the project root (about 45 minutes on this machine; step 1 alone takes 5–8):
```bash
mkdir -p intermediate
for s in 1_load_merge 2_drop_timestamp 3_remove_inf_nan_null 4_remove_corrupt_rows 5_remove_duplicates 6_remove_conflicting_labels 7_fix_dtypes 8_shuffle_rows 9_save_final_csv 10_validate_final_dataset; do python -u codes_for_cleaning/$s.py >> intermediate/pipeline_run.log 2>&1 || break; done
```
- Each step needs the previous step's Parquet file. Those were deleted after the final run, so a change to any step means re-running from step 1. Step 10 alone can still be re-run on the final CSV, because its JSON inputs were kept.
- Run long jobs in the background.
- Test changes on a small sample first (first 20k rows of each raw file in a scratch copy of the folder). On a sample, "All 15 classes present" fails as expected.
- The rule in step 4 and the checks in step 10 (`MINUS_ONE_ALLOWED`, `TIMING_COLS`, `FLOW_TIMEOUT_US`) must stay in sync.

## Current results

| Step | Rows dropped | Rows left |
|---|---|---|
| raw | – | 16,233,002 |
| 1 stray header rows | 59 | 16,232,943 |
| 3 inf/NaN (`Flow Byts/s`, `Flow Pkts/s`) | 95,760 | 16,137,183 |
| 4 corrupt rows (all Benign) | 15 | 16,137,168 |
| 5 exact duplicates | 4,221,045 | 11,916,123 |
| 6 conflicting labels (21,768 feature vectors; 21,743 are Benign vs Infilteration) | 43,541 | **11,872,582** |

The final file, `cicids2018_full_cleaned.csv`, has **11,872,582 rows × 79 columns** (~4.8 GB), with rows shuffled.
- **Data types:** 54 int64, 24 float64, and `Label` as text (`intermediate/7_final_dtypes.json`).
- **Step 10 result:** 26 PASS, 0 FAIL, 0 WARN, 1 INFO.

Final class counts:

| Class | Rows | Class | Rows |
|---|---|---|---|
| Benign | 10,543,008 | Slowloris | 9,908 |
| LOIC-HTTP | 575,364 | LOIC-UDP | 1,730 |
| HOIC | 198,861 | BF-Web | 547 |
| Hulk | 145,199 | BF-XSS | 226 |
| Bot | 144,535 | SQL Injection | 77 |
| Infilteration | 117,598 | SlowHTTPTest | **43** |
| SSH | 94,041 | FTP-BruteForce | **39** |
| GoldenEye | 41,406 | | |

## Things to know when modelling
- **Removing duplicates left two classes tiny.** Without `Timestamp`, repeated attack flows are identical, so step 5 cut FTP-BruteForce from 193,354 to 39 rows and SlowHTTPTest from 139,890 to 43. The user chose to remove all duplicates anyway.
  - Other duplicate losses: HOIC 71%, Hulk 69%, Bot 50%, SSH 50%, Benign 21%.
  - SQL Injection, BF-XSS, SlowHTTPTest and FTP each have fewer than 250 rows. Consider merging or dropping them per model; ask the user.
- **8 columns are all zero** and kept on purpose: `Bwd PSH Flags`, `Bwd URG Flags`, `Fwd/Bwd Byts/b Avg`, `Fwd/Bwd Pkts/b Avg`, `Fwd/Bwd Blk Rate Avg`. They're harmless to most models. Step 10 reports them as INFO.
- **Heavy class imbalance:** Benign is 88.8% of rows. Use a stratified train/test split, and apply class weights or resampling to the training part only.
- **Do these on the training split, not in the dataset:** scaling, label encoding (XGBoost needs integer labels; sklearn accepts text) and resampling. Doing them before the split leaks test information into training.

## Models

### `xg_bruteforce/`: SSH brute-force detector
- **Script:** `train_ssh_bruteforce.py`, run from the project root. It takes about 2.5 minutes.
- **Task:** binary XGBoost, Benign (0) vs SSH-Bruteforce (1). Every other attack is ignored, as the user asked.
- **Data:** all 94,041 SSH rows plus a random 1,000,000 Benign rows (`BENIGN_SAMPLE`). The full Benign set doesn't fit in memory. The CSV is read in chunks as float32.
- **Setup:**
  - Features: 70, after dropping columns that are constant in training.
  - Split: stratified 80/20 train/test, with 10% of train used for early stopping.
  - Class balance: `scale_pos_weight` of about 10.6.
- **Early stopping watches logloss, which is the last item in `eval_metric`.** With `aucpr` last, training stopped at 2 trees, because aucpr is 1.0 from the first round.
- **Result on the 218,731-row test set:** 100% accuracy, precision and recall. 0 false positives, 0 missed attacks, 534 trees.
  - The top features are `Fwd Act Data Pkts`, `Fwd Header Len` and `Dst Port`. The CICIDS2018 SSH brute-force flows are very uniform, so a perfect score here doesn't mean the model will work as well on other networks.
- **Outputs:**
  - `ssh_bruteforce_xgb.json`: the model.
  - `model_features.json`: the feature order and threshold.
  - `evaluation_report.txt`, `confusion_matrix.png`, `feature_importance.png`.
- `xgboost` 3.4.1 was installed with the WMI-safe pip wrapper.

## Git
- Remote: `github.com/samiksha99004/all_10_csv_manual`, branch `main`.
- HEAD holds an **older pipeline** (`merging.py`, XGBoost/IsolationForest models, `merged*.csv`, train/test splits). All of those files are deleted in the working tree, and the current files are untracked.
- Don't commit, restore or push without asking. The CSV and Parquet files are several GB each.
