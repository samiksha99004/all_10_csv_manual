import pandas as pd
import numpy as np
import os
import glob

INPUT_DIR = r"F:\all_10_csv_manual"
INPUT_FILE = os.path.join(INPUT_DIR, "merged_6.csv")
OUTPUT_FILE = os.path.join(INPUT_DIR, "merged_final.csv")
TEMP_DIR = os.path.join(INPUT_DIR, "shuffle_temp")
CHUNK_SIZE = 300_000
N_SHARDS = 20
SEED = 42

os.makedirs(TEMP_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# --- Pass 1: distribute rows randomly into N_SHARDS temp files ---
shard_writers_open = {i: False for i in range(N_SHARDS)}
total_rows = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    total_rows += len(chunk)

    shard_ids = rng.integers(0, N_SHARDS, size=len(chunk))
    chunk["_shard"] = shard_ids

    for shard_id in range(N_SHARDS):
        shard_chunk = chunk[chunk["_shard"] == shard_id].drop(columns=["_shard"])
        if len(shard_chunk) == 0:
            continue
        shard_path = os.path.join(TEMP_DIR, f"shard_{shard_id}.csv")
        shard_chunk.to_csv(
            shard_path,
            mode="a" if shard_writers_open[shard_id] else "w",
            header=not shard_writers_open[shard_id],
            index=False
        )
        shard_writers_open[shard_id] = True

print(f"Total rows distributed into {N_SHARDS} shards: {total_rows}")

# --- Pass 2: shuffle each shard in memory, then append in random shard order ---
shard_order = list(range(N_SHARDS))
rng.shuffle(shard_order)

first_write = True
rows_written = 0

for shard_id in shard_order:
    shard_path = os.path.join(TEMP_DIR, f"shard_{shard_id}.csv")
    if not os.path.exists(shard_path):
        continue
    shard_df = pd.read_csv(shard_path, low_memory=False)
    shard_df = shard_df.sample(frac=1, random_state=SEED + shard_id).reset_index(drop=True)

    shard_df.to_csv(
        OUTPUT_FILE,
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )
    first_write = False
    rows_written += len(shard_df)

print(f"Total rows written to final shuffled file: {rows_written}")
print(f"Saved to: {OUTPUT_FILE}")

# --- Cleanup temp shards ---
for f in glob.glob(os.path.join(TEMP_DIR, "*.csv")):
    os.remove(f)
os.rmdir(TEMP_DIR)
print("Temp shard files cleaned up.")