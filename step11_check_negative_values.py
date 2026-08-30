import pandas as pd
import numpy as np
import os

INPUT_DIR = r"F:\all_10_csv_manual"
INPUT_FILE = os.path.join(INPUT_DIR, "merged_4.csv")
CHUNK_SIZE = 300_000

neg_counts = {}
total_rows = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
    total_rows += len(chunk)

    for col in numeric_cols:
        neg_mask = chunk[col] < 0
        count = neg_mask.sum()
        if count > 0:
            neg_counts[col] = neg_counts.get(col, 0) + count

print(f"Total rows scanned: {total_rows}\n")
print("Columns with negative values (potential data errors):\n")

if not neg_counts:
    print("None found — no invalid negative values.")
else:
    for col, count in sorted(neg_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_rows) * 100
        print(f"{col:<25} {count:>10}  ({pct:.4f}%)")
