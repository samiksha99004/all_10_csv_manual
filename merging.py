import pandas as pd
import glob
import os

INPUT_DIR = r"F:\all_10_csv_manual"
OUTPUT_FILE = os.path.join(INPUT_DIR, "merged.csv")
CHUNK_SIZE = 300_000  # rows per chunk, adjust based on available RAM

csv_files = sorted(f for f in glob.glob(os.path.join(INPUT_DIR, "*.csv"))
                    if os.path.basename(f) != "merged.csv")
print(f"Found {len(csv_files)} files")

reference_cols = None
total_rows = 0

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out:
    for f in csv_files:
        file_rows = 0
        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False):
            chunk.columns = chunk.columns.str.strip()

            if reference_cols is None:
                reference_cols = list(chunk.columns)
                chunk.to_csv(out, header=True, index=False)
            else:
                if list(chunk.columns) != reference_cols:
                    chunk = chunk.reindex(columns=reference_cols)
                chunk.to_csv(out, header=False, index=False)

            file_rows += len(chunk)

        print(f"{os.path.basename(f)}: {file_rows} rows")
        total_rows += file_rows

print(f"\nTotal rows merged: {total_rows}")
print(f"Saved to: {OUTPUT_FILE}")
