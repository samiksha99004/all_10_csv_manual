import pandas as pd
import os

INPUT_DIR = r"F:\all_10_csv_manual"
INPUT_FILE = os.path.join(INPUT_DIR, "merged_4.csv")
CHUNK_SIZE = 300_000

label_counts = {}
total_rows = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False, usecols=["Label"]):
    counts = chunk["Label"].value_counts()
    total_rows += len(chunk)
    for label, count in counts.items():
        label_counts[label] = label_counts.get(label, 0) + count

print(f"Total rows: {total_rows}")
print(f"Total unique labels: {len(label_counts)}\n")

for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    pct = (count / total_rows) * 100
    print(f"{label:<30} {count:>12}  ({pct:.3f}%)")
