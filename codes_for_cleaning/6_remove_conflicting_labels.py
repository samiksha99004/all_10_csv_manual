"""Step 6 - Remove rows whose 78 feature values also appear with a different Label.

After step 5 every (features, Label) row is unique, so any feature vector that
still occurs more than once carries 2+ different labels (mostly Benign vs
Infilteration). No model can separate them, so ALL rows of such vectors are
removed. Every remaining feature vector then has exactly one label.

Pass 1 hashes the features of every row. Pass 2 writes the rows to keep.
"""
from common import STEP5_OUT, STEP6_OUT, iter_chunks, ParquetSink, log_step, Timer  # keep first: disables the hanging WMI query

from collections import Counter

import numpy as np
import pandas as pd

with Timer():
    feat_hashes, codes = [], []
    label_ids = {}           # label text -> small int (keeps 12M labels cheap in memory)
    for chunk in iter_chunks(STEP5_OUT):
        feat_hashes.append(pd.util.hash_pandas_object(chunk.drop(columns=["Label"]), index=False).to_numpy())
        for name in chunk["Label"].unique():
            label_ids.setdefault(name, len(label_ids))
        codes.append(chunk["Label"].map(label_ids).to_numpy(np.int8))
    feat_hashes = np.concatenate(feat_hashes)
    codes = np.concatenate(codes)
    names = {i: name for name, i in label_ids.items()}

    order = np.argsort(feat_hashes, kind="stable")
    fh, lb = feat_hashes[order], codes[order]
    del order
    conflict = (fh[1:] == fh[:-1]) & (lb[1:] != lb[:-1])
    conflict_hashes = np.unique(fh[1:][conflict])
    pairs = Counter(tuple(sorted((names[a], names[b])))
                    for a, b in zip(lb[:-1][conflict].tolist(), lb[1:][conflict].tolist()))
    del fh, lb, conflict
    remove = np.isin(feat_hashes, conflict_hashes)
    del feat_hashes

    before = pd.Series(codes).map(names).value_counts()
    removed = pd.Series(codes[remove]).map(names).value_counts()
    table = pd.DataFrame({"before": before, "removed": removed}).fillna(0).astype("int64")
    table["after"] = table["before"] - table["removed"]
    table["% removed"] = (table["removed"] / table["before"] * 100).round(2)
    print(f"Conflicting feature vectors: {conflict_hashes.size:,}  rows removed: {int(remove.sum()):,}")
    print("Label pairs:")
    for (a, b), n in pairs.most_common():
        print(f"    {a} vs {b}: {n:,}")
    print()
    print(table.sort_values("before", ascending=False).to_string())

    offset = 0
    with ParquetSink(STEP6_OUT) as sink:
        for chunk in iter_chunks(STEP5_OUT):
            n = len(chunk)
            sink.write(chunk[~remove[offset:offset + n]])
            offset += n

    log_step("6_remove_conflicting_labels", len(codes), sink.rows,
             conflicting_feature_vectors=int(conflict_hashes.size),
             removed_by_class={k: int(v) for k, v in removed.items()},
             label_pairs={f"{a} vs {b}": n for (a, b), n in pairs.most_common()})
    print(f"Saved -> {STEP6_OUT}")
