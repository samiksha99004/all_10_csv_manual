import pandas as pd
import os

INPUT_DIR = r"F:\all_10_csv_manual"
INPUT_FILE = os.path.join(INPUT_DIR, "merged_5.csv")
OUTPUT_FILE = os.path.join(INPUT_DIR, "merged_6.csv")
CHUNK_SIZE = 300_000

# collect all unique Protocol values first, so one-hot columns are consistent across chunks
protocol_values = set()
for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False, usecols=["Protocol"]):
    protocol_values.update(chunk["Protocol"].unique())

protocol_values = sorted(protocol_values)
print(f"Unique Protocol values found: {protocol_values}")

first_write = True
total_rows = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()

    # one-hot encode Protocol with fixed categories (consistent columns every chunk)
    chunk["Protocol"] = pd.Categorical(chunk["Protocol"], categories=protocol_values)
    dummies = pd.get_dummies(chunk["Protocol"], prefix="Protocol")
    chunk = pd.concat([chunk.drop(columns=["Protocol"]), dummies], axis=1)

    chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )
    first_write = False
    total_rows += len(chunk)

print(f"Total rows processed: {total_rows}")
print(f"Saved to: {OUTPUT_FILE}")