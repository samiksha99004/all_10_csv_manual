"""Step 10 - Independent cross-check that the final dataset is clean and ML-ready.

Re-reads original_csv/cicids2018_full_cleaned.csv straight from disk (NOT the
intermediate Parquet files). pandas' NA handling is switched off, so empty
cells and text such as "NaN", "null", "None" or "Infinity" are seen exactly as
written instead of being silently turned into NaN.

Each check reports:
  FAIL - something the cleaning must guarantee is wrong: wrong columns,
         inf, NaN/null/empty cells, text in numeric columns, stray header rows,
         bad labels, corrupt rows, duplicate rows, identical features with
         different labels, a column whose data type is not the one fixed in
         step 7, values that overflow float32, column names that ML libraries
         reject, rows that are not shuffled, row/class counts that don't match
         the pipeline records.
  WARN - a data-quality issue in the raw CICIDS2018 data that the cleaning
         does not remove (e.g. flag columns outside 0/1).
  INFO - worth knowing, not a problem (e.g. all-zero columns kept on purpose).
  PASS - no problems found.

The report is printed and saved to cleaning_dataset/10_validation_report.txt.
Exit code is 1 if any check FAILs.
"""
from common import (FINAL_CSV, DTYPES_JSON, SUMMARY_JSON, LOG_JSON, VALIDATION_TXT,
                    EXTRA_ID_COLUMNS, source_csvs, Timer)  # keep first: disables the hanging WMI query

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

REPORT_TXT = VALIDATION_TXT
# XGBoost rejects [ ] <, LightGBM rejects JSON special characters
BAD_NAME_CHARS = set('[]<>{}",:')
FLOAT32_MAX = float(np.finfo(np.float32).max)
# Smaller than CHUNK_ROWS: each chunk is held as text + several float copies
VALIDATE_CHUNK_ROWS = 100_000

KNOWN_LABELS = {
    "Benign", "Bot", "Brute Force -Web", "Brute Force -XSS", "DDOS attack-HOIC",
    "DDOS attack-LOIC-UDP", "DDoS attacks-LOIC-HTTP", "DoS attacks-GoldenEye",
    "DoS attacks-Hulk", "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris",
    "FTP-BruteForce", "Infilteration", "SQL Injection", "SSH-Bruteforce",
}
FLAG_COLS = ["Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
             "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
             "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt"]
# Corrupt-row rule (must match step 4)
MINUS_ONE_ALLOWED = ["Init Fwd Win Byts", "Init Bwd Win Byts"]
FLOW_TIMEOUT_US = 120_000_000
TIMING_COLS = ["Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
               "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
               "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
               "Active Mean", "Active Std", "Active Max", "Active Min",
               "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]
NULL_TOKENS = {"", "nan", "-nan", "+nan", "null", "none", "na", "n/a", "#n/a", "#na", "<na>",
               "nat", "-1.#ind", "1.#ind", "-1.#qnan", "1.#qnan", "#n/a n/a"}

results = []


def check(name, ok, detail="", level="FAIL"):
    results.append(("PASS" if ok else level, name, detail))


def nonzero(series):
    s = series[series > 0]
    return ", ".join(f"{k}={int(v):,}" for k, v in s.items())


with Timer():
    # ---------------------------------------------------------------- schema
    # Expected columns come from the raw files: the standard 80 minus Timestamp
    raw_headers = {}
    for path in source_csvs():
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        raw_headers[os.path.basename(path)] = [c for c in cols if c not in EXTRA_ID_COLUMNS]
    standard = next(iter(raw_headers.values()))
    check("All 10 raw files share the same 80 columns (after Tuesday's 4 ID columns)",
          all(h == standard for h in raw_headers.values()) and len(standard) == 80)
    expected = [c for c in standard if c != "Timestamp"]

    header = pd.read_csv(FINAL_CSV, nrows=0).columns.tolist()
    missing = [c for c in expected if c not in header]
    extra = [c for c in header if c not in expected]
    check(f"Columns match the expected {len(expected)} (raw columns minus Timestamp, same order)",
          header == expected,
          f"missing={missing} extra={extra}" if (missing or extra) else "same names, different order")
    check("No duplicate column names", len(set(header)) == len(header))
    bad_names = [c for c in header if c != c.strip() or any(ch in c for ch in BAD_NAME_CHARS)]
    check("Column names have no leading/trailing spaces or characters XGBoost/LightGBM reject",
          not bad_names, str(bad_names))
    if missing or extra:
        for status, name, detail in results:
            print(f"[{status}] {name} {detail}")
        sys.exit("Schema is wrong - fix that before running the value checks.")

    numeric = [c for c in header if c != "Label"]
    with open(DTYPES_JSON, encoding="utf-8") as f:
        target_dtypes = json.load(f)

    # ---------------------------------------------------------- accumulators
    zeros = pd.Series(0, index=numeric, dtype="int64")
    rows = 0
    stray_header_rows = 0
    null_tokens, text_tokens = zeros.copy(), zeros.copy()
    text_examples = defaultdict(Counter)
    nan_cnt, pos_inf, neg_inf = zeros.copy(), zeros.copy(), zeros.copy()
    negative, below_minus_one, over_timeout = zeros.copy(), zeros.copy(), zeros.copy()
    flag_bad = pd.Series(0, index=FLAG_COLS, dtype="int64")
    col_min = pd.Series(np.inf, index=numeric)
    col_max = pd.Series(-np.inf, index=numeric)
    any_nonzero = pd.Series(False, index=numeric)
    float32_overflow = zeros.copy()
    first_chunk_labels = None
    dtypes_seen = defaultdict(set)
    label_empty = label_space = label_unknown = 0
    unknown_labels = Counter()
    label_counts = Counter()
    row_hashes, feat_hashes, labels = [], [], []
    label_ids = {}           # label text -> small int (keeps 16M labels cheap in memory)

    # ------------------------------------------------------------ main pass
    print(f"Reading {FINAL_CSV} ...")
    reader = pd.read_csv(FINAL_CSV, chunksize=VALIDATE_CHUNK_ROWS, keep_default_na=False, na_values=[],
                         skip_blank_lines=False, low_memory=False)
    for i, chunk in enumerate(reader):
        rows += len(chunk)
        lab = chunk["Label"].astype(str)
        stray_header_rows += int((lab.str.strip() == "Label").sum())

        conv = {}
        for c in numeric:
            s = chunk[c]
            dtypes_seen[c].add("int64" if pd.api.types.is_integer_dtype(s)
                               else "float64" if pd.api.types.is_float_dtype(s) else "text")
            if not pd.api.types.is_numeric_dtype(s):
                n = pd.to_numeric(s, errors="coerce")
                bad = s[n.isna()].astype(str).str.strip()
                bad = bad[bad != c]                     # header tokens counted above
                is_null = bad.str.lower().isin(NULL_TOKENS)
                null_tokens[c] += int(is_null.sum())
                text_tokens[c] += int((~is_null).sum())
                for tok, cnt in bad[~is_null].value_counts().head(5).items():
                    text_examples[c][tok] += int(cnt)
                s = n
            conv[c] = s.astype("float64")
        x = pd.DataFrame(conv)
        dtypes_seen["Label"].add("text" if not pd.api.types.is_numeric_dtype(chunk["Label"]) else "number")

        nan_cnt += x.isna().sum()
        pos_inf += (x == np.inf).sum()
        float32_overflow += (x.abs() > FLOAT32_MAX).sum()
        neg_inf += (x == -np.inf).sum()
        finite = x.where(np.isfinite(x))
        col_min = np.fmin(col_min, finite.min())
        col_max = np.fmax(col_max, finite.max())
        any_nonzero |= (finite.fillna(0) != 0).any()
        negative += (x < 0).sum()
        below_minus_one[MINUS_ONE_ALLOWED] += (x[MINUS_ONE_ALLOWED] < -1).sum()
        over_timeout[TIMING_COLS] += (x[TIMING_COLS] > FLOW_TIMEOUT_US).sum()
        flags = x[FLAG_COLS]
        flag_bad += (flags.notna() & ~flags.isin([0, 1])).sum()

        label_empty += int((lab.str.strip() == "").sum())
        label_space += int((lab != lab.str.strip()).sum())
        unknown = ~lab.isin(KNOWN_LABELS) & (lab.str.strip() != "") & (lab.str.strip() != "Label")
        label_unknown += int(unknown.sum())
        unknown_labels.update(lab[unknown].value_counts().head(5).to_dict())
        label_counts.update(lab.value_counts().to_dict())
        if first_chunk_labels is None:
            first_chunk_labels = lab.value_counts()

        # floats so int/float chunks hash alike
        row_hashes.append(pd.util.hash_pandas_object(x.assign(Label=lab), index=False).to_numpy())
        feat_hashes.append(pd.util.hash_pandas_object(x, index=False).to_numpy())
        for name in lab.unique():
            label_ids.setdefault(name, len(label_ids))
        labels.append(lab.map(label_ids).to_numpy(np.int16))

        if (i + 1) % 20 == 0:
            print(f"  checked {rows:,} rows")
    print(f"  checked {rows:,} rows - done")

    # ------------------------------------------------ duplicates / conflicts
    print("Checking duplicates ...")
    row_hashes = np.concatenate(row_hashes)
    dup_rows = rows - np.unique(row_hashes).size
    del row_hashes

    feat_hashes = np.concatenate(feat_hashes)
    labels = np.concatenate(labels)
    uniques = {i: name for name, i in label_ids.items()}
    order = np.argsort(feat_hashes, kind="stable")
    fh, lb = feat_hashes[order], labels[order]
    del order
    same = fh[1:] == fh[:-1]
    conflict = same & (lb[1:] != lb[:-1])
    conflict_hashes = np.unique(fh[1:][conflict])
    conflict_rows = int(np.isin(feat_hashes, conflict_hashes).sum())
    pair_counts = Counter(zip(np.minimum(lb[:-1], lb[1:])[conflict].tolist(),
                              np.maximum(lb[:-1], lb[1:])[conflict].tolist()))
    del fh, lb, feat_hashes, labels, same, conflict

    # ----------------------------------------------------------- the checks
    check("No stray header rows inside the data", stray_header_rows == 0, f"{stray_header_rows:,} rows")
    check("No empty / null / NaN text cells", null_tokens.sum() == 0, nonzero(null_tokens))
    ex = "; ".join(f"{c}: {dict(v.most_common(3))}" for c, v in text_examples.items())
    check("No non-numeric text in numeric columns", text_tokens.sum() == 0,
          f"{nonzero(text_tokens)}  examples: {ex}" if ex else "")
    check("No NaN values (after numeric conversion)", nan_cnt.sum() == 0, nonzero(nan_cnt))
    check("No +inf values", pos_inf.sum() == 0, nonzero(pos_inf))
    check("No -inf values", neg_inf.sum() == 0, nonzero(neg_inf))
    check("Label is never empty", label_empty == 0, f"{label_empty:,} rows")
    check("Label has no leading/trailing spaces", label_space == 0, f"{label_space:,} rows")
    check("Label only holds the 15 known CICIDS2018 classes", label_unknown == 0,
          f"{label_unknown:,} rows, e.g. {dict(unknown_labels.most_common(3))}")
    absent = sorted(KNOWN_LABELS - set(label_counts))
    check("All 15 classes present", not absent, f"missing: {absent}")

    wrong_type = {c: sorted(s) for c, s in dtypes_seen.items()
                  if c != "Label" and s != {target_dtypes[c]}}
    check("Every numeric column has one data type, the one fixed in step 7", not wrong_type,
          "; ".join(f"{c}: expected {target_dtypes[c]}, found {s}" for c, s in wrong_type.items()))
    check("Label is text in every chunk", dtypes_seen["Label"] == {"text"}, str(dtypes_seen["Label"]))

    neg_report = negative.drop(MINUS_ONE_ALLOWED).add(below_minus_one[MINUS_ONE_ALLOWED], fill_value=0)
    check("No corrupt rows: negative values (Init Win Byts may be -1)", neg_report.sum() == 0,
          nonzero(neg_report))
    check(f"No corrupt rows: timing columns <= {FLOW_TIMEOUT_US:,} us (120 s flow timeout)",
          over_timeout.sum() == 0, nonzero(over_timeout))
    check("No exact duplicate rows (all 79 columns)", dup_rows == 0,
          f"{dup_rows:,} duplicate rows ({dup_rows / rows * 100:.2f}%)")
    pairs = ", ".join(f"{uniques[a]} vs {uniques[b]}: {n:,}" for (a, b), n in pair_counts.most_common(5))
    check("No identical feature rows with different labels", conflict_hashes.size == 0,
          f"{conflict_hashes.size:,} feature vectors / {conflict_rows:,} rows. Top pairs: {pairs}")
    check("All values fit in float32 (sklearn/XGBoost convert to float32)", float32_overflow.sum() == 0,
          nonzero(float32_overflow))
    # Shuffled: every class common enough to expect >= 20 rows in the first chunk must appear there
    n_first = int(first_chunk_labels.sum())
    expected_first = {k: v / rows * n_first for k, v in label_counts.items()}
    not_mixed = sorted(k for k, e in expected_first.items() if e >= 20 and first_chunk_labels.get(k, 0) == 0)
    check(f"Rows are shuffled (first {n_first:,} rows contain every common class)", not not_mixed,
          f"missing from the first rows: {not_mixed}")
    check("Flag columns only 0 or 1", flag_bad.sum() == 0, nonzero(flag_bad), level="WARN")
    all_zero = [c for c in numeric if not any_nonzero[c]]
    check("No column is entirely 0", not all_zero,
          f"{len(all_zero)} all-zero columns kept on purpose (all columns requested): {all_zero}",
          level="INFO")

    # ------------------------------------------ cross-check pipeline outputs
    with open(LOG_JSON, encoding="utf-8") as f:
        expected_rows = list(json.load(f).values())[-1]["rows_out"]
    check("Row count matches cleaning_log.json", rows == expected_rows,
          f"CSV has {rows:,}, log says {expected_rows:,}")
    with open(SUMMARY_JSON, encoding="utf-8") as f:
        summary = json.load(f)
    check("Class counts match step 9 summary", dict(label_counts) == summary["label_counts"],
          str({k: (label_counts.get(k), summary["label_counts"].get(k))
               for k in set(label_counts) | set(summary["label_counts"])
               if label_counts.get(k) != summary["label_counts"].get(k)}))
    mm_diff = [c for c in numeric
               if not (np.isclose(col_min[c], summary["min"][c]) and np.isclose(col_max[c], summary["max"][c]))]
    check("Min/max match step 9 summary (CSV round-trip is lossless)", not mm_diff, str(mm_diff))

    # ---------------------------------------------------------------- report
    lines = ["=" * 100, f"VALIDATION REPORT - {os.path.basename(FINAL_CSV)}",
             f"Rows: {rows:,}   Columns: {len(header)}", "=" * 100]
    for status, name, detail in results:
        lines.append(f"[{status}] {name}")
        if detail and status != "PASS":
            lines.append(f"         -> {detail}")
    counts = Counter(s for s, _, _ in results)
    lines.append("-" * 100)
    lines.append(f"PASS: {counts['PASS']}   FAIL: {counts['FAIL']}   WARN: {counts['WARN']}   INFO: {counts['INFO']}")

    lines.append("\nPer-column summary:")
    lines.append(f"   {'column':<20}{'dtype':<9}{'min':>16}{'max':>16}{'NaN':>6}{'inf':>6}{'text':>6}")
    for c in numeric:
        lines.append(f"   {c:<20}{'/'.join(sorted(dtypes_seen[c])):<9}{col_min[c]:>16.6g}{col_max[c]:>16.6g}"
                     f"{int(nan_cnt[c]):>6,}{int(pos_inf[c] + neg_inf[c]):>6,}"
                     f"{int(null_tokens[c] + text_tokens[c]):>6,}")
    lines.append(f"   {'Label':<20}{'text':<9}")
    lines.append("\nClass distribution:")
    for k, v in sorted(label_counts.items(), key=lambda t: -t[1]):
        lines.append(f"   {k:<28}{v:>12,}")

    text = "\n".join(lines)
    print("\n" + text)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nReport saved -> {REPORT_TXT}")

if counts["FAIL"]:
    sys.exit(1)
