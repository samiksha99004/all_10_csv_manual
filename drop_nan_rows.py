import pandas as pd
import os

INPUT_DIR = r"F:\all_10_csv_manual"
INPUT_FILE = os.path.join(INPUT_DIR, "merged_3.csv")
OUTPUT_FILE = os.path.join(INPUT_DIR, "merged_4.csv")
CHUNK_SIZE = 300_000

first_write = True
rows_before = 0
rows_after = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    rows_before += len(chunk)

    clean_chunk = chunk.dropna()
    rows_after += len(clean_chunk)

    clean_chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )
    first_write = False

print(f"Rows before: {rows_before}")
print(f"Rows after dropping NaN: {rows_after}")
print(f"Rows dropped: {rows_before - rows_after}")
print(f"Saved to: {OUTPUT_FILE}")
