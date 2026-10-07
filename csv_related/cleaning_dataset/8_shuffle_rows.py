"""Step 8 - Shuffle the rows (fixed seed, reproducible).

The merged data is grouped by day file, so reading the first N rows or
splitting without shuffling would give only a few classes. This shuffles all
rows uniformly without loading the dataset into memory:
  pass 1 - send every row to one of ~120 random bucket files
  pass 2 - load each bucket (~100k rows), shuffle it in memory, append it
A random bucket followed by a random order inside each bucket is a uniform
random permutation. Bucket files are deleted afterwards.
"""
from common import STEP7_OUT, STEP8_OUT, WORK_DIR, CHUNK_ROWS, iter_chunks, ParquetSink, log_step, Timer  # keep first: disables the hanging WMI query

import math
import os
import shutil

import numpy as np
import pyarrow.parquet as pq

SEED = 42

with Timer():
    rng = np.random.default_rng(SEED)
    rows_in = pq.ParquetFile(STEP7_OUT).metadata.num_rows
    n_buckets = max(1, math.ceil(rows_in / CHUNK_ROWS))
    bucket_dir = os.path.join(WORK_DIR, "8_shuffle_buckets")
    shutil.rmtree(bucket_dir, ignore_errors=True)
    os.makedirs(bucket_dir)

    sinks = [ParquetSink(os.path.join(bucket_dir, f"bucket_{b:03d}.parquet")) for b in range(n_buckets)]
    try:
        for chunk in iter_chunks(STEP7_OUT):
            bucket = rng.integers(0, n_buckets, size=len(chunk))
            for b in range(n_buckets):
                part = chunk[bucket == b]
                if len(part):
                    sinks[b].write(part)
    finally:
        for s in sinks:
            s.close()
    print(f"Pass 1: {rows_in:,} rows spread over {n_buckets} random buckets")

    with ParquetSink(STEP8_OUT) as sink:
        for b in range(n_buckets):
            path = os.path.join(bucket_dir, f"bucket_{b:03d}.parquet")
            if not os.path.exists(path):
                continue
            part = pq.read_table(path, use_threads=False).to_pandas()
            sink.write(part.iloc[rng.permutation(len(part))])
    shutil.rmtree(bucket_dir)
    print(f"Pass 2: buckets shuffled and joined -> {sink.rows:,} rows")

    log_step("8_shuffle_rows", rows_in, sink.rows, seed=SEED, buckets=n_buckets)
    print(f"Saved -> {STEP8_OUT}")
