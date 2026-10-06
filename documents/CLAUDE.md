# CICIDS2018 – merge & clean pipeline

This folder merges the 10 daily CICIDS2018 CSV files (CICFlowMeter network-flow features) into one cleaned, **ML-ready** dataset. The output keeps **all 79 columns**: the 80 raw columns minus `Timestamp`. It has:
- no NaN, null, empty or ±inf values;
- no corrupt rows and no duplicate rows;
- no identical features carrying different labels;
- one correct data type per column;
- rows shuffled.

**Status:** cleaning is done, and one model is trained: the 15-class all-attacks classifier (`1_xgboost_model/`). The user decides the next steps.

## What the user wants (keep to this)
- **Clean only; never drop columns.** No feature selection, no removal of zero or correlated columns, unless the user asks for it. An earlier version dropped columns (46 left) and the user rejected it.
- The final CSV has **79 columns**: the raw columns minus `Timestamp`. There is no `source_file` column and no `time` feature. `Label` keeps the **original text** (not encoded).
- `original_csv/` must hold exactly **11 CSVs**: the 10 raw files plus `cicids2018_full_cleaned.csv`. The large temporary Parquet files go in `intermediate/`, which git ignores. The small cleaning records live in `cleaning_dataset/`.
- **Folder layout (set by the user; keep it):** only `README.md`, `CLAUDE.md` and `.gitignore` sit at the root, because GitHub, Claude Code and git only read them there. Everything else lives in `cleaning_dataset/`, `1_xgboost_model/`, `isolation_forest_and_xgboost_model/`, `notebooks/` or `original_csv/`.
- One script per step, named `N_stepname.py`.
- The user confirmed these removals: all exact duplicates (even though that leaves FTP-BruteForce and SlowHTTPTest with only a few dozen rows), all rows whose features appear with more than one label, and the 15 corrupt rows. Ask before removing anything else.

## Folder layout
```
all_10_csv_manual/
├── CLAUDE.md, README.md, .gitignore        must stay at the root (README = GitHub front page)
├── model_comparison_report.pdf             the three models compared side by side
├── conference_paper.pdf                    write-up (added by the user)
├── ssh_plan.txt                            plan to live-test 2_xgboost_model against a real SSH brute force
├── cleaning_dataset/
│   ├── common.py                           paths (data folders one level up), chunked Parquet I/O, log_step, WMI workaround
│   ├── 1_load_merge.py … 10_validate_final_dataset.py
│   ├── cleaning_log.json, 7_final_dtypes.json, 9_final_summary.json, 10_validation_report.txt   records of the run
│   └── cleaning_dataset_report.pdf         formatted report of the cleaning pipeline (source, steps, columns, classes)
├── 1_xgboost_model/                       XGBoost 15-class model: Benign + 14 attack types (see "Models" below)
│   ├── train_all_attacks.py, tune_class_weights.py, evaluate_final.py
│   ├── all_attacks_xgb.json + label/feature/weight JSONs, metrics, confusion matrices, plots, logs
│   └── training_report.pdf             training, tree storage, data types, testing, metrics, confusion matrix
├── 2_xgboost_model/                        two-level XGBoost: level 1 attack/not -> level 2 which attack (96.3%)
│   ├── 1_prepare_data.py, 2_train_level1.py, 3_train_level2.py, 4_evaluate.py, xgb_common.py, live_detect.py
│   ├── level1_binary_xgb.json, level2_multiclass_xgb.json, metrics, reports, steps.txt
│   └── training_report.pdf
├── isolation_forest_and_xgboost_model/      hybrid: Isolation Forest (normal/attack) -> XGBoost (which attack)
│   ├── if_common.py, 1_prepare_data.py, 2_train_isolation_forest.py, 3_evaluate_isolation_forest.py,
│   │   4_train_xgboost.py, 5_evaluate_hybrid.py
│   ├── steps.txt                           how it is trained and how to run it (the user runs it, not Claude)
│   └── data/                               train/test arrays from step 1 (~650 MB, git-ignored)
├── notebooks/
│   └── cicids2018-using-decision-trees.ipynb   reference notebook (Kaggle, uses 1 file); not part of the pipeline
├── original_csv/                           10 raw day files + cicids2018_full_cleaned.csv (git-ignored)
└── intermediate/                           created only while the pipeline runs: per-step Parquet files (git-ignored)
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
  - For ad-hoc snippets run from the project root, first do `sys.path.insert(0, "cleaning_dataset")`, then `import common`.
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
Each step streams the previous step's Parquet file from `intermediate/` and writes a new one. It also records rows in and out in `cleaning_dataset/cleaning_log.json` through `log_step()`; re-running a step overwrites its entry. Numeric columns stay float64 in the intermediate files, and step 6 sets the final types.

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
for s in 1_load_merge 2_drop_timestamp 3_remove_inf_nan_null 4_remove_corrupt_rows 5_remove_duplicates 6_remove_conflicting_labels 7_fix_dtypes 8_shuffle_rows 9_save_final_csv 10_validate_final_dataset; do python -u cleaning_dataset/$s.py >> intermediate/pipeline_run.log 2>&1 || break; done
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
- **Data types:** 54 int64, 24 float64, and `Label` as text (`cleaning_dataset/7_final_dtypes.json`).
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

## Live detection (requirement set 2026-10-05; the project is being built toward this)
- **Goal:** while an attack runs, print every 2-3 seconds whether the traffic is NORMAL or ABNORMAL. When ABNORMAL, print a guess table of which attack it might be.
- **Architecture (set 2026-10-05, current):** the VICTIM does everything: capture, conversion, feature selection, scoring and the terminal table. Windows is not part of the live pipeline, and the attacker only attacks. The victim needs a copy of the 2-level model files (`live_detection/victim/models/`) and of the Java converter (`CICFlowMeter-4.0`, one-time copy from the attacker). The monitor is `live_detection/victim/live_monitor.py`. It prints the attacker IP (the source with the most port-22 flows), a NORMAL/ABNORMAL verdict per 2-second window, a guess table for that window, and cumulative counts, redrawn every 2 seconds.
- **Disk retention (victim monitor):** keep the newest 40 pcaps and their CSVs. When more than 60 pcaps exist, delete the oldest until 40 remain. A backlog of more than 60 unconverted pcaps loses the oldest ones, and the screen reports the count deleted.
- **Earlier shared-folder version (superseded):** `1_capture_rotate.sh`, `2_convert_loop.sh` and `3_live_watch.py` used the Windows shared folder and are kept only as history.
- **Earlier pipeline (superseded, for reference):**
  1. `1_capture_rotate.sh` (victim): tcpdump writes a new pcap every 2 seconds (`-G 2`) into the shared folder, filtered to the attacker's IP and port 22.
  2. `2_convert_loop.sh` (attacker): converts each finished 2-second pcap with the Java CICFlowMeter-v3 into a flow CSV. The newest pcap is skipped because tcpdump may still be writing it.
  3. `3_live_watch.py` (Windows): picks up each new flow CSV, maps the Java columns to the 78 training features (microseconds, no scaling), scores it with a chosen model, and prints the verdict and guess table.
- **Verdict rule:** ABNORMAL if at least one flow in the window is flagged as attack by level 1 (2-level model) or by the chosen model. NORMAL otherwise. Windows with few flows are noisy; the rule is written so it can be changed.
- **Limits to report with every result:** 2-second windows hold only a few flows, so percentages per window are noisy. Each window is converted separately, so flows that cross a window boundary are split.
- **Status:** the three scripts are written. They have NOT been tested end-to-end on live traffic. The watcher is tested on a saved CSV only.
- **Attack procedure (order matters):** start the watcher, start the converter loop, start the victim capture, then run Patator from the attacker. Stop in reverse order.

## Models

- `xgboost` 3.4.1 was installed with the WMI-safe pip wrapper.
- XGBoost early stopping follows the **last** metric in `eval_metric`. A metric that is perfect from round 1 (such as `aucpr` on an easy task) stops training almost immediately, so put `logloss` or `mlogloss` last.
- The user had a separate binary SSH brute-force model (`xg_bruteforce/`), its training PDF and a Kali/Patator lab-attack procedure PDF. **All of these were deleted on request.** Don't recreate them unless asked.

### `1_xgboost_model/`: 15-class attack classifier
- **Task:** label each flow as Benign or one of the 14 attack types. The user's target is at least 95% recall for every class and at least 95% overall accuracy.
- **Data:** every attack row plus a random 1,000,000 Benign rows (2,329,184 rows). Split 80/20 per class (stratified), with 10% of the training part held out as validation.
- **Scripts** (run from the project root):
  1. `train_all_attacks.py` (about 17 minutes): XGBoost `multi:softprob`, depth 10, learning rate 0.1, up to 500 trees (497 used), 70 features. Uses square-root "balanced" sample weights. Early stopping uses the unweighted validation loss.
  2. `tune_class_weights.py` (about 3 minutes): searches one decision weight per class on the validation set only; prediction is `argmax(probability * weight)`. Weights are saved in `class_weights.json`. `prepare_splits()` in the training script makes all three scripts use the same split.
  3. `evaluate_final.py` (about 2 minutes): the final test. It predicts the 465,837 test rows with the tuned weights and writes `final_metrics.json` (overall and per-class precision/recall/F1/TP/FN/FP), plus `confusion_matrix_final.csv` and `.png`. Prediction speed is about 56,800 flows per second.
- **Things that failed, don't repeat them:** full "balanced" weights plus early stopping on the weighted validation loss stopped at 22 trees. That run reached 85.5% overall and dropped Benign recall to 68%.
- **Test result after tuning:** overall accuracy 95.70%, and 10 of 15 classes reach 95% recall. Below 95%:
  - Infilteration 51.8% (23,520 test rows): its flows look like normal traffic, and it is the class most confused with Benign.
  - Brute Force -XSS 93.3% (45 test rows), SQL Injection 73.3% (15), FTP-BruteForce 25.0% (8), DoS attacks-SlowHTTPTest 0% (9). Removing duplicates left these classes with too few rows to learn or test reliably.
- **Open decision for the user:** merge the tiny classes into families (for example Web attack = BF-Web + BF-XSS + SQL), drop them from this model, or keep the 15 classes. Nothing is merged yet.
- **Outputs:** `all_attacks_xgb.json` (55.6 MB, under GitHub's 100 MB limit), `label_classes.json`, `model_features.json`, `class_weights.json`, `evaluation_report.txt`, `evaluation_report_tuned.txt`, `final_metrics.json`, `confusion_matrix_final.csv`, `confusion_matrix.png` (untuned), `confusion_matrix_final.png` (tuned, with counts), `per_class_recall.png`, `per_class_recall_tuned.png`, and the run logs.
- **Model storage** (explained in `training_report.pdf`): the model is JSON with 7,500 trees (500 rounds × 15 classes). `tree_info` gives each tree's class. Prediction uses rounds 0–497, which is 7,470 trees. Each tree is stored as parallel arrays per node (`left_children`, `right_children`, `split_indices`, `split_conditions`, `default_left`, …). The trees have 876,836 nodes and 442,168 leaves in total, with a median depth of 7 and a maximum of 10.
- **Error pattern:** Benign ↔ Infilteration accounts for 19,743 of the 20,031 test mistakes. There are 288 other mistakes.
- **Report:** `1_xgboost_model/training_report.pdf`. It was built by a reportlab script kept outside the repo, and every number in it comes from `final_metrics.json`, `class_weights.json`, `label_classes.json` and `cleaning_dataset/9_final_summary.json`.

### `2_xgboost_model/`: two-level XGBoost (96.3% offline)
- **Live-test status (2026-10-05, updated):** five live SSH brute-force attempts, see `ssh_plan.txt`. Only attempt 3 (Java CICFlowMeter-v3, Ubuntu victim) gave usable naming: SSH-Bruteforce 29.0%, Infilteration 44.7%, Slowloris 16.0%, Benign 10.2%. Attempts 4-5 (Hydra, Kali-to-Kali, Python `cicflowmeter`) gave 0% recall. Their packet counts match training (18-24 vs 22), but flows are ~46-52x too long because each packet gap is ~47-53x training's.
- **Column alignment (verified):** Python `cicflowmeter` writes snake_case names and time columns in **seconds**; training uses microseconds, so the 23 time columns are multiplied by 1e6. Round-trip test: 5,000 training test rows written in Python format and read back gave zero feature error and 100% prediction agreement. The Python converter's flag and window values still differ from training on the same SSH traffic, so its 0% results are not a fair read of the model.
- **Input features (decided 2026-10-05):** the saved models use 78 features (`data/meta.json`). 70 are used by tree splits in both models. 8 are never split on: Bwd PSH Flags, Bwd URG Flags, Fwd Byts/b Avg, Fwd Pkts/b Avg, Fwd Blk Rate Avg, Bwd Byts/b Avg, Bwd Pkts/b Avg, Bwd Blk Rate Avg. Keep all 78 columns in the input, set the 8 to 0. A 70-column input fails, and a 70-feature model needs retraining. Equivalence check run: all 465,837 test rows predicted with the 8 columns original, zero, and random (1-1000); final labels identical on every row (0 differ), so they cannot change the output.
- **Live detection script:** `2_xgboost_model/live_detect.py` (replay and follow modes, console output only). Replay tested on the Kali file; follow mode not yet tested.
- **Design:** level 1 is binary — Benign vs not-Benign (attack/not). Only flows level 1 calls "attack" go to level 2, which outputs a probability for each of the 14 attack classes; the model's answer is whichever class has the highest probability, i.e. level 2 is a percentage table per flow, not a single fixed guess.
- **Live SSH brute-force test (2026-10-05):** real Patator attack, Kali -> Ubuntu VM, captured and converted with the real Java CICFlowMeter-v3, 792 confirmed attack flows (`ssh_attack.pcap_Flow.csv`, kept local, not committed). Two earlier attempts (Python cicflowmeter tool, different `MaxAuthTries` settings) both gave 0% recall from flow-shape mismatches in opposite directions; see `ssh_plan.txt` for the full history.
  - Level 1 flagged 711/792 (89.8%) as attack (the "something is wrong" gate still works reasonably well).
  - Level 2 named them: Infilteration 44.7%, SSH-Bruteforce 29.0% (correct), Slowloris 16.0%, Benign 10.2% (missed at the gate).
  - **Overall recall on this live capture: 29.0%**, vs 100% SSH-Bruteforce recall on the offline 20% test split. Cause: live flows are much shorter/burstier than training (duration median 2,073us vs ~400,000us; ~3 bwd packets vs ~22) — closer in shape to Slowloris/Infilteration than to the SSH-Bruteforce flows the model was trained on. This looks like a genuine 2018-testbed-vs-live-LAN difference, not a tunable setting (3 attempts, 3 flow shapes, same outcome direction on naming).
  - No PDF/txt written for this test per user request; numbers above are the full record.

### `isolation_forest_and_xgboost_model/`: hybrid Isolation Forest + XGBoost
- **The user runs these scripts; Claude must not run them.** `steps.txt` holds only the run commands.
- **The code was rebuilt on 2026-09-29 after the first run failed. The earlier results are void; the user must re-run steps 2–5.** The first version flagged 94.8% of normal traffic as attack (59% accuracy) because it had no feature scaling and its threshold rule maximised class coverage while ignoring Benign false alarms.
- **The fix (from the IDS Isolation Forest literature):**
  1. features are `log1p`-compressed **and standardised** with a `StandardScaler` fitted on the Benign training rows (the single biggest lever);
  2. `contamination=0.02` (low), `max_samples=4096`, `n_estimators=300`;
  3. the threshold is the point of maximum balanced accuracy (Youden's J) between normal and attack, so stage-1 accuracy is the best this Isolation Forest can give. The report also gives ROC-AUC (threshold-free separation) so the ceiling is visible. The user chose to keep the Isolation Forest for stage 1 (over a supervised binary model) knowing it caps stage-1 accuracy on this data.
- **Design:** stage 1 (Isolation Forest) says normal vs attack; flows it flags go to stage 2 (XGBoost) which names the attack or answers Benign to cancel a false alarm; flows called normal are final Benign. Both models share the 80/20 split, so test rows are unseen by both.
- **Data / memory:** all attack rows + ~1M Benign, stratified 80/20, no validation set. Step 1 reads the CSV in 200k-row chunks to `data/*.npy`; later steps memory-map and score in chunks.
- **Stage 2 (step 4):** XGBoost on the same split, all 15 classes, raw (unscaled) features, settings from `1_xgboost_model/` (500 trees, depth 10, lr 0.1, sqrt-balanced weights), fixed 500 trees, no weight tuning.
- **Honest expectation:** standardising should lift normal accuracy well above 95%, but the dense DoS/DDoS floods score like normal traffic, so at a 5% false-alarm ceiling the Isolation Forest may still catch few of them. The user's target (>= 95% on both normal AND attack for every class) may not be reachable from pure unsupervised Isolation Forest on this data; the test report will show the real trade-off. Naming the attack still relies on the XGBoost stage.
- **Outputs after the user runs it:**
  - Stage 1: `isolation_forest.joblib`, `model_config.json`, `train_report.txt`, `test_report.txt`, `test_metrics.json`, `per_attack_recall.png`, `score_distribution.png`.
  - Stage 2: `xgboost_stage2.json`, `train_xgboost_report.txt`.
  - Hybrid: `hybrid_report.txt`, `hybrid_metrics.json`, `hybrid_confusion_matrix.csv/.png`.

## Git
- Remote: `github.com/samiksha99004/all_10_csv_manual`, branch `main`.
- **Standing rule (user request): after every step, update this CLAUDE.md to reflect the change, then commit and push to `main`.** This is durable authorization to push without asking each time.
- **Exception:** the hybrid work (renaming the folder to `isolation_forest_and_xgboost_model/`, steps 4–5, steps.txt and the doc updates) was **not committed or pushed**, because the user said not to. Wait for the user before pushing it.
- The repo now mirrors the working tree, except that `.gitignore` excludes the CSV and Parquet files (over GitHub's 100 MB limit), so pushes carry only code, docs, the small JSON records and results, the trained model and the PDF reports. The old pipeline files (`merging.py`, `.pkl` models, old `step*.py`) were removed from the repo; they remain in earlier history. `README.md` is the GitHub front page; update it when results or scripts change.
- Auth: the stored Git Credential Manager credential authenticates as `samiksha99004`. Collaborators with write access: `shraddhamaria25`, `shreshta-del` (owner/admin: `samiksha99004`).
