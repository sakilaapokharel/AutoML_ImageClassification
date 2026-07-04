"""Dataset selection + loading. Corresponds to the 'Dataset' -> 'Dataset loaders'
boxes in the diagram.

Supports two layouts under data/<name>/, auto-detected:

A) CSV layout (used whenever train.csv + test.csv exist):

    data/emotions/
        images_train/...          (images, any nesting -- e.g. images_train/imgs/*.jpg)
        images_test/...           (images, any nesting -- e.g. images_test/imgsss/*.jpg)
        train.csv                 (a filename column + a label column)
        test.csv                  (same columns, same label vocabulary as train.csv)

    Filename/label column names are auto-detected from common conventions
    (filename/image/id/... and label/class/emotion/...). If detection fails,
    pass filename_col / label_col explicitly to load_dataset().

    Image lookup is done by filename against a recursive index of the image
    folder, so it doesn't matter how deeply the actual image files are nested
    (images_train/imgs/xxx.jpg, images_train/xxx.jpg, etc. all work), and the
    csv's filename value can be with or without extension.

"""

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torchvision import datasets

IMG_SIZE = 224
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")

FILENAME_CANDIDATES = ["image_file_name"]
LABEL_CANDIDATES = ["label"]


def list_available_datasets(data_root="data"):
    """Return the names of all dataset folders under data_root, e.g. ['emotions', ...]."""
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root '{data_root}' does not exist")
    return sorted(
        d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
    )


def _guess_column(df, candidates, role):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not auto-detect the {role} column among {list(df.columns)}. "
        f"Pass filename_col=/label_col= explicitly to load_dataset()."
    )


def _build_file_index(root_dir):
    """Recursively index every image file under root_dir by both its full
    filename and its extension-less stem, so csv rows can reference images at
    any nesting depth and with or without a file extension."""
    index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.lower().endswith(IMAGE_EXTS):
                continue
            full_path = os.path.join(dirpath, fn)
            index.setdefault(fn, full_path)
            index.setdefault(os.path.splitext(fn)[0], full_path)
    return index


class CSVImageDataset(torch.utils.data.Dataset):
    """(image, label) dataset driven by a CSV file + an image directory
    (possibly nested)."""

    def __init__(
        self,
        image_dir,
        csv_path,
        filename_col="image_file_name",
        label_col="label",
        class_to_idx=None,
    ):
        df = pd.read_csv(csv_path)
        self.filename_col = filename_col or _guess_column(
            df, FILENAME_CANDIDATES, "filename"
        )
        self.label_col = label_col or _guess_column(df, LABEL_CANDIDATES, "label")

        self.filenames = df[self.filename_col].astype(str).tolist()
        raw_labels = df[self.label_col].astype(str).tolist()

        if class_to_idx is None:
            classes = sorted(set(raw_labels))
            class_to_idx = {c: i for i, c in enumerate(classes)}
        unknown = sorted(set(raw_labels) - set(class_to_idx))
        if unknown:
            raise ValueError(
                f"{csv_path} contains label(s) {unknown} not seen in the training "
                f"set's class list {sorted(class_to_idx)}."
            )
        self.class_to_idx = class_to_idx
        self.classes = sorted(class_to_idx, key=lambda c: class_to_idx[c])
        self.targets = [class_to_idx[l] for l in raw_labels]

        self._file_index = _build_file_index(image_dir)
        if not self._file_index:
            raise FileNotFoundError(f"No image files found under {image_dir}")

    def __len__(self):
        return len(self.filenames)

    def _resolve_path(self, fname):
        if fname in self._file_index:
            return self._file_index[fname]
        stem = os.path.splitext(fname)[0]
        if stem in self._file_index:
            return self._file_index[stem]
        raise FileNotFoundError(
            f"Could not find image '{fname}' anywhere under the image directory "
            f"(checked {len(self._file_index)} indexed files)."
        )

    def __getitem__(self, idx):
        path = self._resolve_path(self.filenames[idx])
        img = Image.open(path)
        if img.mode not in ["RGB", "L"]:
            img = img.convert("RGB")
        return img, self.targets[idx]


@dataclass
class DatasetBundle:
    train: torch.utils.data.Dataset
    test: torch.utils.data.Dataset
    classes: list
    num_channels: int

def _find_existing(base, candidates):
    for c in candidates:
        p = os.path.join(base, c)
        if os.path.exists(p):
            return p
    return None

def _detect_num_channels(dataset):
    sample_img, _ = dataset[0]

    if sample_img.mode == "RGB":
        return 3
    elif sample_img.mode == "L":
        return 1
    else:
        return len(sample_img.getbands())

def load_dataset(
    name,
    data_root="data",
    filename_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> DatasetBundle:
    """Load a dataset by folder name, auto-detecting layout A (CSV) vs
    layout B (ImageFolder) as described in the module docstring."""
    base = os.path.join(data_root, name)
    if not os.path.isdir(base):
        available = list_available_datasets(data_root)
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {base}. Available datasets: {available}"
        )

    train_csv = _find_existing(base, ["train.csv"])
    test_csv = _find_existing(base, ["test.csv"])

    if train_csv and test_csv:
        train_dir = _find_existing(base, ["images_train"]) or base
        test_dir = _find_existing(base, ["images_test"]) or base

        train_ds = CSVImageDataset(train_dir, train_csv, filename_col, label_col)
        test_ds = CSVImageDataset(
            test_dir,
            test_csv,
            filename_col,
            label_col,
            class_to_idx=train_ds.class_to_idx,
        )
        num_channels = _detect_num_channels(train_ds)
        return DatasetBundle(train=train_ds, test=test_ds, classes=train_ds.classes, num_channels=num_channels)
