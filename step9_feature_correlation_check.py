import pandas as pd
import numpy as np
import os

INPUT_DIR = r"F:\all_10_csv_manual"
INPUT_FILE = os.path.join(INPUT_DIR, "merged_4.csv")
CHUNK_SIZE = 300_000
CORR_THRESHOLD = 0.95

# --- Pass 1: accumulate sums for correlation matrix across chunks ---
numeric_cols = None
n = 0
sum_x = None
sum_x2 = None
sum_xy = None

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()

    if numeric_cols is None:
        numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
        sum_x = np.zeros(len(numeric_cols))
        sum_x2 = np.zeros(len(numeric_cols))
        sum_xy = np.zeros((len(numeric_cols), len(numeric_cols)))

    X = chunk[numeric_cols].to_numpy(dtype=np.float64)
    n += X.shape[0]
    sum_x += X.sum(axis=0)
    sum_x2 += (X ** 2).sum(axis=0)
    sum_xy += X.T @ X

# --- Compute correlation matrix from accumulated sums ---
mean = sum_x / n
var = (sum_x2 / n) - mean ** 2
std = np.sqrt(var)

cov = (sum_xy / n) - np.outer(mean, mean)
denom = np.outer(std, std)
denom[denom == 0] = np.nan
corr = cov / denom

corr_df = pd.DataFrame(corr, index=numeric_cols, columns=numeric_cols)

# --- Find highly correlated pairs ---
to_drop = set()
pairs = []
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        c = corr_df.iloc[i, j]
        if pd.notna(c) and abs(c) >= CORR_THRESHOLD:
            col_i, col_j = numeric_cols[i], numeric_cols[j]
            pairs.append((col_i, col_j, c))
            # drop the second of the pair by convention
            to_drop.add(col_j)

print(f"Highly correlated pairs (|corr| >= {CORR_THRESHOLD}):")
for col_i, col_j, c in pairs:
    print(f"  {col_i} <-> {col_j}: {c:.4f}")

print(f"\nCandidate columns to drop: {len(to_drop)}")
for c in sorted(to_drop):
    print(f"- {c}")

print(f"\nRemaining feature count if dropped: {len(numeric_cols) - len(to_drop)}")
