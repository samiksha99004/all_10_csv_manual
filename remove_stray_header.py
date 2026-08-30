import pandas as pd
import os

INPUT_DIR = r"F:\all_10_csv_manual"
TARGET_FILE = os.path.join(INPUT_DIR, "merged_4.csv")
TEMP_FILE = os.path.join(INPUT_DIR, "merged_4_temp.csv")
CHUNK_SIZE = 300_000

first_write = True
rows_before = 0
rows_after = 0

for chunk in pd.read_csv(TARGET_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    rows_before += len(chunk)

    # drop any row where Label column literally equals "Label" (stray header row)
    clean_chunk = chunk[chunk["Label"] != "Label"]
    rows_after += len(clean_chunk)

    clean_chunk.to_csv(
        TEMP_FILE,
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )
    first_write = False

os.replace(TEMP_FILE, TARGET_FILE)

print(f"Rows before: {rows_before}")
print(f"Rows after removing stray header row(s): {rows_after}")
print(f"Rows removed: {rows_before - rows_after}")
print(f"merged_4.csv updated in place at: {TARGET_FILE}")
