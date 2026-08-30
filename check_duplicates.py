import pandas as pd
import os

INPUT_DIR = r"F:\all_10_csv_manual"
FILE_TO_CHECK = os.path.join(INPUT_DIR, "merged_2.csv")
CHUNK_SIZE = 300_000

seen = set()
total_rows = 0
duplicate_count = 0

for chunk in pd.read_csv(FILE_TO_CHECK, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    row_hashes = chunk.apply(lambda r: hash(tuple(r)), axis=1)

    for h in row_hashes:
        total_rows += 1
        if h in seen:
            duplicate_count += 1
        else:
            seen.add(h)

print(f"Total rows scanned: {total_rows}")
print(f"Duplicate rows found: {duplicate_count}")
print("CLEAN - no duplicates" if duplicate_count == 0 else "DUPLICATES STILL PRESENT")