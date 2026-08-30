import pandas as pd
import numpy as np
import os

INPUT_DIR = r"F:\all_10_csv_manual"
INPUT_FILE = os.path.join(INPUT_DIR, "merged_3.csv")
CHUNK_SIZE = 300_000

min_vals = {}
max_vals = {}
total_rows = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    numeric_cols = chunk.select_dtypes(include=[np.number]).columns

    total_rows += len(chunk)

    for col in numeric_cols:
        c_min = chunk[col].min(skipna=True)
        c_max = chunk[col].max(skipna=True)

        if col not in min_vals:
            min_vals[col] = c_min
            max_vals[col] = c_max
        else:
            min_vals[col] = min(min_vals[col], c_min)
            max_vals[col] = max(max_vals[col], c_max)

print(f"Total rows scanned: {total_rows}\n")
print("Columns that are ALL ZERO (min == max == 0):\n")

zero_cols = []
for col in min_vals:
    if min_vals[col] == 0 and max_vals[col] == 0:
        zero_cols.append(col)
        print(f"- {col}")

if not zero_cols:
    print("None found - no column is all zero.")

print(f"\nTotal all-zero columns: {len(zero_cols)}")
