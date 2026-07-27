import pandas as pd
from pathlib import Path

dataset_name = "emotions"

train_path = Path("data") / dataset_name / "train.csv"

df = pd.read_csv(train_path)

print("Columns:")
print(df.columns)

# Class distribution
counts = df["label"].value_counts().sort_index()

print("\nClass distribution:")
print(counts)

print("\nPercentages:")
print((counts / len(df) * 100).round(2))