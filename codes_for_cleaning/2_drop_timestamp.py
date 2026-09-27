"""Step 2 - Drop the Timestamp column (80 -> 79 columns)."""
from common import STEP1_OUT, STEP2_OUT, iter_chunks, ParquetSink, log_step, Timer  # keep first: disables the hanging WMI query

with Timer():
    rows_in = 0
    with ParquetSink(STEP2_OUT) as sink:
        for chunk in iter_chunks(STEP1_OUT):
            rows_in += len(chunk)
            sink.write(chunk.drop(columns=["Timestamp"]))

    print(f"Columns now: {len(sink.schema.names)}")
    log_step("2_drop_timestamp", rows_in, sink.rows, dropped_columns=["Timestamp"])
    print(f"Saved -> {STEP2_OUT}")
