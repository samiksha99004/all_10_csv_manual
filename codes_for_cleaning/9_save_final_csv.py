"""Step 9 - Save the final cleaned dataset as original_csv/cicids2018_full_cleaned.csv
and print a summary: final shape, class distribution (value_counts on Label),
data type + min/max per column, and rows dropped at each cleaning step.

The summary is also saved to intermediate/9_final_summary.json for step 10.
"""
from common import (STEP8_OUT, DTYPES_JSON, SUMMARY_JSON, LOG_JSON, FINAL_CSV, CSV_DIR,
                    iter_chunks, parquet_columns, Timer)  # keep first: disables the hanging WMI query

import json
import os
from collections import Counter

import numpy as np
import pandas as pd

with Timer():
    cols = parquet_columns(STEP8_OUT)
    with open(DTYPES_JSON, encoding="utf-8") as f:
        dtypes = json.load(f)
    numeric = [c for c in cols if c != "Label"]

    rows = 0
    col_min = pd.Series(np.inf, index=numeric)
    col_max = pd.Series(-np.inf, index=numeric)
    label_counts = Counter()

    tmp = FINAL_CSV + ".partial"
    with open(tmp, "w", encoding="utf-8", newline="") as out:
        for i, chunk in enumerate(iter_chunks(STEP8_OUT)):
            chunk.to_csv(out, index=False, header=(i == 0))
            rows += len(chunk)
            col_min = np.fmin(col_min, chunk[numeric].min())
            col_max = np.fmax(col_max, chunk[numeric].max())
            label_counts.update(chunk["Label"].value_counts().to_dict())
    os.replace(tmp, FINAL_CSV)
    print(f"Saved -> {FINAL_CSV} ({os.path.getsize(FINAL_CSV) / 1e9:.2f} GB)")

    print(f"\nFinal shape: ({rows:,}, {len(cols)})")

    vc = pd.Series(label_counts).sort_values(ascending=False)
    print("\nClass distribution (df['Label'].value_counts()):")
    print(pd.DataFrame({"count": vc, "percent": (vc / rows * 100).round(4)}).to_string())

    print("\nData type and range of each column:")
    print(f"   {'column':<20}{'dtype':<9}{'min':>18}{'max':>18}")
    for c in cols:
        if c == "Label":
            print(f"   {c:<20}{'str':<9}{'(text)':>18}")
        else:
            print(f"   {c:<20}{dtypes[c]:<9}{col_min[c]:>18.8g}{col_max[c]:>18.8g}")

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump({"shape": [rows, len(cols)], "columns": cols,
                   "label_counts": {k: int(v) for k, v in vc.items()},
                   "min": {k: float(v) for k, v in col_min.items()},
                   "max": {k: float(v) for k, v in col_max.items()}}, f, indent=2)

    with open(LOG_JSON, encoding="utf-8") as f:
        log = json.load(f)
    print("\n" + "=" * 72)
    print("CLEANING SUMMARY - rows dropped at each step")
    print("=" * 72)
    print(f"{'step':<32}{'rows in':>14}{'rows out':>14}{'dropped':>12}")
    for step, e in log.items():
        print(f"{step:<32}{e['rows_in']:>14,}{e['rows_out']:>14,}{e['rows_dropped']:>12,}")
    first, last = next(iter(log.values())), list(log.values())[-1]
    total = first["rows_in"] - last["rows_out"]
    print("-" * 72)
    print(f"{'TOTAL':<32}{first['rows_in']:>14,}{last['rows_out']:>14,}{total:>12,}"
          f"  ({total / first['rows_in'] * 100:.3f}%)")

    csvs = sorted(f for f in os.listdir(CSV_DIR) if f.lower().endswith(".csv"))
    print(f"\nCSV files in {CSV_DIR}: {len(csvs)}")
    for f in csvs:
        print("  ", f)
