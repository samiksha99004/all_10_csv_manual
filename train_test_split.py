import pandas as pd
import os

INPUT_DIR = r"F:\all_10_csv_manual"
INPUT_FILE = os.path.join(INPUT_DIR, "merged_final.csv")
TRAIN_FILE = os.path.join(INPUT_DIR, "train_dataset.csv")
TEST_FILE = os.path.join(INPUT_DIR, "test_dataset.csv")

TRAIN_FRACTION = 0.8
RANDOM_SEED = 42
CHUNK_SIZE = 300_000

first_write_train = True
first_write_test = True
train_rows = 0
test_rows = 0
total_rows = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    total_rows += len(chunk)

    # random 80/20 split within each chunk (dataset should already be shuffled globally)
    train_chunk = chunk.sample(frac=TRAIN_FRACTION, random_state=RANDOM_SEED)
    test_chunk = chunk.drop(train_chunk.index)

    train_chunk.to_csv(
        TRAIN_FILE,
        mode="w" if first_write_train else "a",
        header=first_write_train,
        index=False
    )
    test_chunk.to_csv(
        TEST_FILE,
        mode="w" if first_write_test else "a",
        header=first_write_test,
        index=False
    )
    first_write_train = False
    first_write_test = False

    train_rows += len(train_chunk)
    test_rows += len(test_chunk)

print(f"Total rows: {total_rows}")
print(f"Train rows: {train_rows} ({train_rows/total_rows*100:.2f}%)")
print(f"Test rows: {test_rows} ({test_rows/total_rows*100:.2f}%)")
print(f"\nSaved train set to: {TRAIN_FILE}")
print(f"Saved test set to: {TEST_FILE}")
