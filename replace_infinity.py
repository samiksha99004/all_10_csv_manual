import pandas as pd
import numpy as np
import os

INPUT_DIR = r"F:\all_10_csv_manual"
TARGET_FILE = os.path.join(INPUT_DIR, "merged_3.csv")
TEMP_FILE = os.path.join(INPUT_DIR, "merged_3_temp.csv")
CHUNK_SIZE = 300_000

first_write = True
total_rows = 0
total_inf_replaced = 0

for chunk in pd.read_csv(TARGET_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()

    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(chunk[numeric_cols])
    total_inf_replaced += inf_mask.values.sum()

    chunk[numeric_cols] = chunk[numeric_cols].replace([np.inf, -np.inf], np.nan)

    chunk.to_csv(
        TEMP_FILE,
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )
    first_write = False
    total_rows += len(chunk)

os.replace(TEMP_FILE, TARGET_FILE)

print(f"Total rows processed: {total_rows}")
print(f"Infinity values replaced with NaN: {total_inf_replaced}")
print(f"merged_3.csv updated in place at: {TARGET_FILE}")
