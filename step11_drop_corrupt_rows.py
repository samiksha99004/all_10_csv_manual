import pandas as pd
import os

INPUT_DIR = r"F:\all_10_csv_manual"
INPUT_FILE = os.path.join(INPUT_DIR, "merged_4.csv")
OUTPUT_FILE = os.path.join(INPUT_DIR, "merged_5.csv")
CHUNK_SIZE = 300_000

# only these columns are true data errors when negative
# (Init Fwd/Bwd Win Byts excluded -- -1 is a valid sentinel there)
CHECK_COLS = [
    "Flow IAT Min", "Fwd IAT Min", "Flow Duration",
    "Flow Pkts/s", "Flow IAT Mean", "Fwd IAT Tot",
    "Fwd IAT Mean", "Flow IAT Max", "Fwd IAT Max"
]

first_write = True
rows_before = 0
rows_after = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    rows_before += len(chunk)

    cols_present = [c for c in CHECK_COLS if c in chunk.columns]
    bad_mask = (chunk[cols_present] < 0).any(axis=1)
    clean_chunk = chunk[~bad_mask]

    rows_after += len(clean_chunk)

    clean_chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )
    first_write = False

print(f"Rows before: {rows_before}")
print(f"Rows after removing corrupt rows: {rows_after}")
print(f"Rows dropped: {rows_before - rows_after}")
print(f"Saved to: {OUTPUT_FILE}")
