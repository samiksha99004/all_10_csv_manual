import pandas as pd
import os

INPUT_DIR = r"F:\all_10_csv_manual"
INPUT_FILE = os.path.join(INPUT_DIR, "merged_2.csv")
OUTPUT_FILE = os.path.join(INPUT_DIR, "merged_3.csv")
CHUNK_SIZE = 300_000

seen = set()
reference_cols = None
first_write = True

rows_in = 0
rows_out = 0

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out:
    for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        if reference_cols is None:
            reference_cols = list(chunk.columns)

        rows_in += len(chunk)

        row_hashes = chunk.apply(lambda r: hash(tuple(r)), axis=1)

        keep_mask = []
        for h in row_hashes:
            if h in seen:
                keep_mask.append(False)
            else:
                seen.add(h)
                keep_mask.append(True)

        deduped_chunk = chunk[keep_mask]
        rows_out += len(deduped_chunk)

        deduped_chunk.to_csv(out, header=first_write, index=False)
        first_write = False

        print(f"Processed {rows_in} rows so far, kept {rows_out}")

print(f"\nTotal rows read: {rows_in}")
print(f"Total rows after dedup: {rows_out}")
print(f"Duplicates removed: {rows_in - rows_out}")
print(f"Saved to: {OUTPUT_FILE}")
