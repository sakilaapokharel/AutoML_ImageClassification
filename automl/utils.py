"""Shared helpers used by multiple pipeline modules."""
import torch

import os
import pandas as pd

def get_targets(dataset):
    """Recursively resolve the integer class-label list for a Dataset/Subset."""
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base = get_targets(dataset.dataset)
        return [base[i] for i in dataset.indices]
    raise AttributeError("Could not resolve targets for dataset of type "
                          f"{type(dataset)}")


class TransformedDataset(torch.utils.data.Dataset):
    """Wraps a (PIL-image, label) dataset and applies a torchvision transform
    lazily at __getitem__ time. Lets us reuse one ImageFolder / Subset across
    many augmentation configs without re-reading images from disk."""

    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label

def merge_all_csv(output_dir):
    all_dfs = []
    all_columns = set()

    # Step 1: collect all CSVs except the merged output file
    csv_files = [
        os.path.join(f"{output_dir}/individual_logs", f)
        for f in os.listdir(output_dir)
    ]

    for file in csv_files:
        df = pd.read_csv(file)
        all_dfs.append(df)
        all_columns.update(df.columns)

    all_columns = list(all_columns)

    # Step 2: align all dataframes to same columns
    aligned_dfs = []
    for df in all_dfs:
        df = df.reindex(columns=all_columns)
        aligned_dfs.append(df)

    # Step 3: concatenate
    final_df = pd.concat(aligned_dfs, ignore_index=True)

    # Step 4: replace NaN with None
    final_df = final_df.where(pd.notnull(final_df), None)

    # Step 5: save
    output_file = os.path.join(output_dir, "combined_metadata_csv.csv")
    final_df.to_csv(output_file, index=False)

    print(f"Merged CSV saved to: {output_file}")

