import pandas as pd
import os

INPUT_DIR = r"F:\all_10_csv_manual"
INPUT_FILE = os.path.join(INPUT_DIR, "merged_1.csv")
OUTPUT_FILE = os.path.join(INPUT_DIR, "merged_2.csv")
CHUNK_SIZE = 300_000

COLUMNS_TO_DROP = [
    "Timestamp",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg",
    "CWE Flag Cnt",
]

first_write = True
total_rows = 0
final_columns = None

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()

    cols_present = [c for c in COLUMNS_TO_DROP if c in chunk.columns]
    chunk = chunk.drop(columns=cols_present)

    if final_columns is None:
        final_columns = list(chunk.columns)

    chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )
    first_write = False
    total_rows += len(chunk)

print(f"Total rows written: {total_rows}")
print(f"Saved to: {OUTPUT_FILE}")

print(f"\nTotal columns: {len(final_columns)}\n")
for i, c in enumerate(final_columns, 1):
    print(f"{i}. {c}")