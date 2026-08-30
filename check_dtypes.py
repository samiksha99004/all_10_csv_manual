import pandas as pd
import os

INPUT_DIR = r"F:\all_10_csv_manual"
INPUT_FILE = os.path.join(INPUT_DIR, "merged_4.csv")

# read a decent-sized sample to infer dtypes accurately
sample = pd.read_csv(INPUT_FILE, nrows=500_000, low_memory=False)
sample.columns = sample.columns.str.strip()

print(f"Total columns: {len(sample.columns)}\n")
print(f"{'Column':<25} {'Dtype':<12}")
print("-" * 40)
for col in sample.columns:
    print(f"{col:<25} {str(sample[col].dtype):<12}")

print("\n--- Unique values in likely-categorical columns ---")
for col in ["Protocol", "Dst Port", "Label"]:
    if col in sample.columns:
        n_unique = sample[col].nunique()
        print(f"\n{col}: {n_unique} unique values")
        if n_unique <= 30:
            print(sample[col].value_counts())
