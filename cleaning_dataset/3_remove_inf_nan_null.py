"""Step 3 - Treat inf / -inf as missing, then drop every row that has an inf,
NaN, null or empty value in any column (including an empty Label).

Works column by column on each chunk (no full-chunk copies) to keep memory low.
"""
from common import STEP2_OUT, STEP3_OUT, iter_chunks, ParquetSink, log_step, Timer  # keep first: disables the hanging WMI query

from collections import Counter

import numpy as np

with Timer():
    rows_in = 0
    inf_counts, missing_counts = Counter(), Counter()
    with ParquetSink(STEP3_OUT) as sink:
        for chunk in iter_chunks(STEP2_OUT):
            rows_in += len(chunk)
            drop = np.zeros(len(chunk), dtype=bool)
            for col in chunk.columns:
                if col == "Label":
                    missing = chunk[col].isna().to_numpy() | (chunk[col].astype(str).str.strip() == "").to_numpy()
                else:
                    v = chunk[col].to_numpy()
                    inf = np.isinf(v)
                    inf_counts[col] += int(inf.sum())
                    missing = np.isnan(v) | inf
                missing_counts[col] += int(missing.sum())
                drop |= missing
            sink.write(chunk[~drop])

    inf_counts = {k: v for k, v in inf_counts.items() if v}
    missing_counts = {k: v for k, v in missing_counts.items() if v}
    print(f"inf/-inf values found: {sum(inf_counts.values()):,}")
    print("\n".join(f"    {k}: {v:,}" for k, v in inf_counts.items()) or "    none")
    print("inf + NaN/null/empty values per column:")
    print("\n".join(f"    {k}: {v:,}" for k, v in missing_counts.items()) or "    none")
    log_step("3_remove_inf_nan_null", rows_in, sink.rows,
             inf_values=sum(inf_counts.values()), inf_per_column=inf_counts,
             missing_or_inf_per_column=missing_counts)
    print(f"Saved -> {STEP3_OUT}")
