from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

FILE_COL = "image_file_name"
LABEL_COL = "label"


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in (FILE_COL, LABEL_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing expected columns: {missing}. Found: {df.columns.tolist()}")
    return df, FILE_COL, LABEL_COL


def sample_per_class(df, label_col, n_per_class, seed=42):
    """Take up to n_per_class rows for every class label."""
    if n_per_class == -1:
        return df.reset_index(drop=True)
    parts = []
    for label, group in df.groupby(label_col):
        take = min(n_per_class, len(group))
        parts.append(group.sample(n=take, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


class ImageCSVDataset(Dataset):
    def __init__(self, df, file_col, label_col, images_dir, transform):
        self.df = df.reset_index(drop=True)
        self.file_col = file_col
        self.label_col = label_col
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, fname):
        p = self.images_dir / fname
        if p.exists():
            return p
        for ext in (".jpg", ".jpeg", ".png"):
            cand = self.images_dir / f"{fname}{ext}"
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Could not find image for '{fname}' in {self.images_dir}")

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = self._resolve_path(str(row[self.file_col]))
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = row[self.label_col]
        return img, label