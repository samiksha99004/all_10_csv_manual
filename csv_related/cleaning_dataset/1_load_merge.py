"""Step 1 - Load all 10 CICIDS2018 CSVs and concatenate them into one dataset.

- Result has the 80 standard columns (Dst Port ... Label), in the raw order.
- Thuesday-20-02-2018 has 4 extra identifier columns (Flow ID, Src IP,
  Src Port, Dst IP) that the other 9 files don't have. They are dropped so all
  files share the same 80 columns (otherwise 9 files would get NaN there).
- Header lines repeated inside the data (59 of them) are removed.
- Numeric columns are stored as float64 in the intermediate files so every
  chunk has the same schema. Final data types are set in step 6.
"""
from common import (STEP1_OUT, CHUNK_ROWS, EXTRA_ID_COLUMNS, source_csvs,
                    ParquetSink, log_step, Timer)  # keep first: disables the hanging WMI query

import os

import numpy as np
import pandas as pd

TEXT_COLUMNS = ["Timestamp", "Label"]

with Timer():
    files = source_csvs()
    print(f"Found {len(files)} CSV files:")
    for f in files:
        print("  ", os.path.basename(f))
    assert len(files) == 10, "Expected exactly 10 source CSV files in original_csv/"

    columns = None
    raw_rows = 0
    stray_headers = 0
    coerced_to_nan = 0
    per_file_rows = {}

    with ParquetSink(STEP1_OUT) as sink:
        for path in files:
            name = os.path.basename(path)
            file_rows = 0
            for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False):
                raw_rows += len(chunk)
                chunk = chunk.drop(columns=[c for c in EXTRA_ID_COLUMNS if c in chunk.columns])

                is_header = chunk["Label"].astype(str).str.strip() == "Label"
                stray_headers += int(is_header.sum())
                chunk = chunk[~is_header]

                if columns is None:
                    columns = list(chunk.columns)
                assert sorted(chunk.columns) == sorted(columns), f"{name}: unexpected columns"
                chunk = chunk[columns]

                for col in columns:
                    if col in TEXT_COLUMNS:
                        continue
                    if not pd.api.types.is_numeric_dtype(chunk[col]):
                        before = int(chunk[col].isna().sum())
                        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                        coerced_to_nan += int(chunk[col].isna().sum()) - before
                    chunk[col] = chunk[col].astype(np.float64)
                for col in TEXT_COLUMNS:
                    chunk[col] = chunk[col].astype(str).where(chunk[col].notna(), None)

                sink.write(chunk)
                file_rows += len(chunk)
            per_file_rows[name] = file_rows
            print(f"  loaded {name}: {file_rows:,} rows")

    print(f"\nMerged shape: ({sink.rows:,}, {len(columns)})")
    print(f"Stray header rows removed: {stray_headers:,}")
    print(f"Non-numeric cells turned into NaN (removed in step 3): {coerced_to_nan:,}")
    log_step("1_load_merge", raw_rows, sink.rows,
             stray_header_rows=stray_headers, coerced_to_nan=coerced_to_nan,
             dropped_columns_tuesday_only=EXTRA_ID_COLUMNS, rows_per_file=per_file_rows)
    print(f"Saved -> {STEP1_OUT}")
