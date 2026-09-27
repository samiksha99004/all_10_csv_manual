"""Step 7 - Give every column one correct data type.

- int64   if every value in the column is a whole number
- float64 otherwise
- Label stays text

The type is decided from the values, not the raw text. Some raw files store
whole numbers in rounded scientific notation (e.g. "1.00E+07" in Flow IAT Max),
so pandas reads the same column as int in one file and float in another.
Pass 1 checks every value. Pass 2 casts. The chosen types are saved to
intermediate/7_final_dtypes.json and checked again in step 10.
"""
from common import STEP6_OUT, STEP7_OUT, DTYPES_JSON, iter_chunks, parquet_columns, ParquetSink, log_step, Timer  # keep first: disables the hanging WMI query

import json

import numpy as np
import pandas as pd

with Timer():
    cols = parquet_columns(STEP6_OUT)
    numeric = [c for c in cols if c != "Label"]

    whole = pd.Series(True, index=numeric)
    for chunk in iter_chunks(STEP6_OUT, columns=numeric):
        whole &= (chunk % 1 == 0).all()

    dtypes = {c: ("int64" if whole[c] else "float64") for c in numeric}
    dtypes["Label"] = "str"
    dtypes = {c: dtypes[c] for c in cols}      # keep column order

    rows_in = 0
    int_cols = [c for c in numeric if dtypes[c] == "int64"]
    with ParquetSink(STEP7_OUT) as sink:
        for chunk in iter_chunks(STEP6_OUT):
            rows_in += len(chunk)
            chunk[int_cols] = chunk[int_cols].astype(np.int64)
            chunk["Label"] = chunk["Label"].astype(str)
            sink.write(chunk)

    with open(DTYPES_JSON, "w", encoding="utf-8") as f:
        json.dump(dtypes, f, indent=2)

    float_cols = [c for c in numeric if dtypes[c] == "float64"]
    print(f"int64   ({len(int_cols)}): {int_cols}")
    print(f"\nfloat64 ({len(float_cols)}): {float_cols}")
    print("\nstr     (1): ['Label']")
    log_step("7_fix_dtypes", rows_in, sink.rows, int64_columns=int_cols, float64_columns=float_cols)
    print(f"Data types -> {DTYPES_JSON}")
    print(f"Saved -> {STEP7_OUT}")
