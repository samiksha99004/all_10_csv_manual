"""Step 5 - Remove exact duplicate rows (identical in all 79 columns, Label
included). The first occurrence is kept.

Pass 1 hashes every row. Pass 2 writes only the first copy of each row.
Rows with identical features but DIFFERENT labels are not duplicates and are
kept.
"""
from common import STEP4_OUT, STEP5_OUT, iter_chunks, ParquetSink, log_step, Timer  # keep first: disables the hanging WMI query

import numpy as np
import pandas as pd

with Timer():
    hashes, codes = [], []
    label_ids = {}           # label text -> small int (keeps 16M labels cheap in memory)
    for chunk in iter_chunks(STEP4_OUT):
        # 64-bit row hash; chance of any false match over ~16M rows is ~1e-5
        hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
        for name in chunk["Label"].unique():
            label_ids.setdefault(name, len(label_ids))
        codes.append(chunk["Label"].map(label_ids).to_numpy(np.int8))
    hashes = np.concatenate(hashes)
    codes = np.concatenate(codes)
    duplicate = pd.Series(hashes).duplicated(keep="first").to_numpy()
    del hashes

    names = {i: name for name, i in label_ids.items()}
    before = pd.Series(codes).map(names).value_counts()
    dups = pd.Series(codes[duplicate]).map(names).value_counts()
    table = pd.DataFrame({"before": before, "duplicates": dups}).fillna(0).astype("int64")
    table["after"] = table["before"] - table["duplicates"]
    table["% removed"] = (table["duplicates"] / table["before"] * 100).round(2)
    print(f"Duplicate rows: {int(duplicate.sum()):,}\n")
    print(table.sort_values("before", ascending=False).to_string())

    offset = 0
    with ParquetSink(STEP5_OUT) as sink:
        for chunk in iter_chunks(STEP4_OUT):
            n = len(chunk)
            sink.write(chunk[~duplicate[offset:offset + n]])
            offset += n

    log_step("5_remove_duplicates", len(codes), sink.rows,
             duplicates_by_class={k: int(v) for k, v in dups.items()})
    print(f"Saved -> {STEP5_OUT}")
