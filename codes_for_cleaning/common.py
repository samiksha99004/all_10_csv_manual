"""Shared paths and chunked I/O helpers for the CICIDS2018 cleaning pipeline.

The merged dataset (~16M rows) does not fit comfortably in RAM, so every step
streams the previous step's Parquet file in chunks and writes a new one.
Intermediate files live in ./intermediate so that ./original_csv ends up with
only the 10 source CSVs + the final cleaned CSV.

Import this module BEFORE pandas/sklearn: on this machine the Windows WMI
service hangs, and Python's platform module queries WMI (via pandas, sklearn
and pip imports), which freezes the process. Disabling the WMI query makes
platform fall back to sys.getwindowsversion().
"""
import platform


def _wmi_disabled(*args, **kwargs):
    raise OSError("WMI query disabled")


platform._wmi_query = _wmi_disabled

import json
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

# Use the system allocator so freed chunk memory goes back to Windows; this
# machine often has only ~3 GB of commit headroom (see iter_chunks).
pa.set_memory_pool(pa.system_memory_pool())

# Scripts live in codes_for_cleaning/; the data folders are one level up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(BASE_DIR, "original_csv")
WORK_DIR = os.path.join(BASE_DIR, "intermediate")
os.makedirs(WORK_DIR, exist_ok=True)

STEP1_OUT = os.path.join(WORK_DIR, "1_merged.parquet")
STEP2_OUT = os.path.join(WORK_DIR, "2_no_timestamp.parquet")
STEP3_OUT = os.path.join(WORK_DIR, "3_no_inf_nan_null.parquet")
STEP4_OUT = os.path.join(WORK_DIR, "4_no_corrupt_rows.parquet")
STEP5_OUT = os.path.join(WORK_DIR, "5_no_duplicates.parquet")
STEP6_OUT = os.path.join(WORK_DIR, "6_no_conflicting_labels.parquet")
STEP7_OUT = os.path.join(WORK_DIR, "7_fixed_dtypes.parquet")
STEP8_OUT = os.path.join(WORK_DIR, "8_shuffled.parquet")
DTYPES_JSON = os.path.join(WORK_DIR, "7_final_dtypes.json")
SUMMARY_JSON = os.path.join(WORK_DIR, "9_final_summary.json")
LOG_JSON = os.path.join(WORK_DIR, "cleaning_log.json")
FINAL_CSV = os.path.join(CSV_DIR, "cicids2018_full_cleaned.csv")

CHUNK_ROWS = 100_000

# Columns only present in Thuesday-20-02-2018 (not in the other 9 files)
EXTRA_ID_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP"]


def source_csvs():
    """The 10 raw CICIDS2018 day files (everything in original_csv/ except the output)."""
    return sorted(os.path.join(CSV_DIR, f) for f in os.listdir(CSV_DIR)
                  if f.lower().endswith(".csv")
                  and os.path.abspath(os.path.join(CSV_DIR, f)) != os.path.abspath(FINAL_CSV))


def iter_chunks(path, columns=None, batch_size=CHUNK_ROWS):
    """Yield pandas DataFrames from a Parquet file, batch_size rows at a time.

    Reads one row group at a time without threads. ParquetFile.iter_batches
    leaks: a full pass over a 12M-row file grew to ~2.8 GB here, versus a flat
    ~650 MB with read_row_group.
    """
    pf = pq.ParquetFile(path)
    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i, columns=columns, use_threads=False)
        for start in range(0, table.num_rows, batch_size):
            yield table.slice(start, batch_size).to_pandas()
        del table


def parquet_columns(path):
    return pq.ParquetFile(path).schema_arrow.names


class ParquetSink:
    """Append DataFrame chunks to one Parquet file with a fixed schema."""

    def __init__(self, path):
        self.path = path
        self.writer = None
        self.schema = None
        self.rows = 0

    def write(self, df):
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.schema = table.schema.remove_metadata()
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="snappy")
        else:
            table = table.cast(self.schema)
        self.writer.write_table(table)
        self.rows += len(df)

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def log_step(step, rows_in, rows_out, **details):
    """Record rows in/out for a step in cleaning_log.json (re-running a step overwrites its entry)."""
    log = {}
    if os.path.exists(LOG_JSON):
        with open(LOG_JSON, encoding="utf-8") as f:
            log = json.load(f)
    log[step] = {"rows_in": rows_in, "rows_out": rows_out,
                 "rows_dropped": rows_in - rows_out, **details}
    with open(LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(log.items())), f, indent=2)
    print(f"\n[{step}] rows in: {rows_in:,}  rows out: {rows_out:,}  dropped: {rows_in - rows_out:,}")


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        print(f"Elapsed: {time.time() - self.t0:.1f}s")
