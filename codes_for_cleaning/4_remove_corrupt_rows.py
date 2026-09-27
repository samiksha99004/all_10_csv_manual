"""Step 4 - Remove corrupt rows: physically impossible flows in the raw data.

A row is corrupt if it has
  - a negative value in any column (Init Fwd/Bwd Win Byts may be -1, which
    CICFlowMeter uses for "not set"), e.g. negative Flow Duration / Flow Pkts/s
  - a timing column above CICFlowMeter's 120 s flow timeout (120,000,000 us)
In the full dataset this is 15 Benign rows.
"""
from common import STEP3_OUT, STEP4_OUT, iter_chunks, ParquetSink, log_step, Timer  # keep first: disables the hanging WMI query

from collections import Counter

import numpy as np

MINUS_ONE_ALLOWED = ["Init Fwd Win Byts", "Init Bwd Win Byts"]
FLOW_TIMEOUT_US = 120_000_000
TIMING_COLS = ["Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
               "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
               "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
               "Active Mean", "Active Std", "Active Max", "Active Min",
               "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]

with Timer():
    rows_in = 0
    by_column, by_class = Counter(), Counter()
    with ParquetSink(STEP4_OUT) as sink:
        for chunk in iter_chunks(STEP3_OUT):
            rows_in += len(chunk)
            bad = np.zeros(len(chunk), dtype=bool)
            for col in chunk.columns:
                if col == "Label":
                    continue
                v = chunk[col].to_numpy()
                m = v < (-1 if col in MINUS_ONE_ALLOWED else 0)
                if col in TIMING_COLS:
                    m |= v > FLOW_TIMEOUT_US
                by_column[col] += int(m.sum())
                bad |= m
            by_class.update(chunk.loc[bad, "Label"].value_counts().to_dict())
            sink.write(chunk[~bad])

    by_column = {k: v for k, v in by_column.most_common() if v}
    print(f"Corrupt rows removed: {rows_in - sink.rows:,}  by class: {dict(by_class)}")
    print("Bad values per column:")
    print("\n".join(f"    {k}: {v:,}" for k, v in by_column.items()) or "    none")
    log_step("4_remove_corrupt_rows", rows_in, sink.rows,
             rule="negative value (Init Win Byts may be -1) or timing column > 120 s",
             bad_values_per_column=by_column, corrupt_by_class=dict(by_class))
    print(f"Saved -> {STEP4_OUT}")
