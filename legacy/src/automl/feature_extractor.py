"""
Meta-feature extraction for the AutoML vision datasets.

Layout this is built for (matches data/<dataset_name>/):
    images_train/000001.jpg, 000002.jpg, ...
    images_test/000001.jpg, ...
    train.csv   -> columns: image_id/filename, label
    test.csv

Design:
    - CSVImageDataset: reads one split (train or test) of one dataset.
      Does not load images into memory eagerly -- iterates lazily.
    - BaseFeatureExtractor: contract every feature group follows.
        extract(dataset) -> dict
    - ClassDistributionFeatures / ImageStatisticsFeatures: two concrete
      feature groups, each independent and unit-testable. Only train-split
      features are computed -- test-derived features (sample counts, ratios)
      were dropped since they didn't add selector signal.
    - MetaFeatureExtractor: orchestrator. Runs every group and merges
      results into one flat dict -- one row per dataset, ready for a
      pandas DataFrame / TabPFN input table.

To add a new feature group later (e.g. embedding-based complexity
features): write a new class with the same extract() signature and
append it to MetaFeatureExtractor's default `groups` list. Nothing
else changes.
"""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------

class CSVImageDataset:
    """
    One split (train or test) of one dataset, backed by a CSV of labels
    plus a folder of images.

    Adjust `id_col` / `label_col` below if your CSV headers differ from
    "image_id" / "label" -- run `head -3 data/<name>/train.csv` to check.
    """

    def __init__(
        self,
        images_dir: str,
        csv_path: str,
        id_col: str = "image_file_name",
        label_col: str = "label",
        name: str = "dataset",
    ):
        self.images_dir = images_dir
        self.df = pd.read_csv(csv_path)
        self.id_col = id_col
        self.label_col = label_col
        self.name = name

        # normalize labels to 0..num_classes-1 integers for feature math
        self._label_codes = self.df[label_col].astype("category").cat.codes.to_numpy()

    @property
    def labels(self) -> np.ndarray:
        return self._label_codes

    def __len__(self) -> int:
        return len(self.df)

    @property
    def num_classes(self) -> int:
        return len(np.unique(self._label_codes))

    def _image_path(self, image_id) -> str:
        # image_id might already include an extension, or might be an int
        # id that needs zero-padding to match "000001.jpg" style filenames.
        s = str(image_id)
        if os.path.splitext(s)[1]:  # already has an extension
            return os.path.join(self.images_dir, s)
        return os.path.join(self.images_dir, f"{int(s):06d}.jpg")

    def iter_images(self) -> Iterator[Tuple[np.ndarray, int]]:
        for row, label in zip(self.df[self.id_col], self._label_codes):
            path = self._image_path(row)
            with Image.open(path) as img:
                yield np.array(img), int(label)

    @property
    def num_channels(self) -> int:
        first_id = self.df[self.id_col].iloc[0]
        with Image.open(self._image_path(first_id)) as img:
            return 1 if img.mode in ("L", "1") else len(img.getbands())


# ---------------------------------------------------------------------------
# Feature extractor contract
# ---------------------------------------------------------------------------

class BaseFeatureExtractor(ABC):
    """Every feature group implements extract(dataset) -> dict."""

    prefix: str = ""

    @abstractmethod
    def extract(self, dataset: CSVImageDataset) -> Dict[str, Any]:
        raise NotImplementedError

    def _prefixed(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if not self.prefix:
            return features
        return {f"{self.prefix}{k}": v for k, v in features.items()}


# ---------------------------------------------------------------------------
# Feature group 1: class distribution / imbalance
# ---------------------------------------------------------------------------

class ClassDistributionFeatures(BaseFeatureExtractor):
    """num_classes, per-class counts, imbalance ratio, distribution entropy."""

    prefix = "class_"

    def extract(self, dataset: CSVImageDataset) -> Dict[str, Any]:
        _, counts = np.unique(dataset.labels, return_counts=True)
        num_classes = len(counts)

        imbalance_ratio = float(counts.max() / counts.min())

        return self._prefixed({
            "num_classes": int(num_classes),
            "avg_images_per_class": float(counts.mean()),
            "min_images_per_class": int(counts.min()),
            "max_images_per_class": int(counts.max()),
            "imbalance_ratio": imbalance_ratio,
        })


# ---------------------------------------------------------------------------
# Feature group 2: image geometry / pixel intensity
# ---------------------------------------------------------------------------

class ImageStatisticsFeatures(BaseFeatureExtractor):
    """
    Resolution and pixel-intensity stats. Pixel-intensity stats are computed
    on a random sample (default 500 images) since reading all 60k fashion
    images just for a mean/std would be wasteful. Width/height are read for
    every image via cheap PIL header reads.
    """

    prefix = "img_"

    def __init__(self, sample_size: int = 500, seed: int = 42):
        self.sample_size = sample_size
        self.seed = seed

    def extract(self, dataset: CSVImageDataset) -> Dict[str, Any]:
        rng = random.Random(self.seed)
        n = len(dataset)
        sample_size = min(self.sample_size, n)
        sample_indices = set(rng.sample(range(n), sample_size))

        widths, heights = [], []
        pixel_means, pixel_stds = [], []

        for idx, (img, _label) in enumerate(dataset.iter_images()):
            arr = np.asarray(img)
            h, w = arr.shape[0], arr.shape[1]
            widths.append(w)
            heights.append(h)
            if idx in sample_indices:
                pixel_means.append(arr.astype(np.float32).mean())
                pixel_stds.append(arr.astype(np.float32).std())

        widths = np.array(widths)
        heights = np.array(heights)

        return self._prefixed({
            "mean_width": float(widths.mean()),
            "mean_height": float(heights.mean()),
            "mean_pixel_value": float(np.mean(pixel_means)),
            "std_pixel_value": float(np.mean(pixel_stds)),
        })


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MetaFeatureExtractor:
    """
    Runs every feature group and merges results into one flat dict per
    dataset. Append new groups to `groups` to extend -- nothing else
    needs to change.
    """

    def __init__(self, groups=None):
        self.groups = groups or [
            ClassDistributionFeatures(),
            ImageStatisticsFeatures(),
        ]

    def extract(
        self,
        train_dataset: CSVImageDataset,
        dataset_name: str = None,
    ) -> Dict[str, Any]:
        features: Dict[str, Any] = {
            "dataset_name": dataset_name or train_dataset.name,
        }
        for group in self.groups:
            features.update(group.extract(train_dataset))

        features["num_train_samples"] = len(train_dataset)
        return features

    def extract_all(self, dataset_specs: Dict[str, dict]) -> pd.DataFrame:
        """
        dataset_specs: {
            "fashion": {"images_train": ..., "train_csv": ...},
            ...
        }
        Only the train split is needed now that test-derived features have
        been dropped. Returns a DataFrame, one row per dataset -- your
        TabPFN input table.
        """
        rows = []
        for name, spec in dataset_specs.items():
            train_ds = CSVImageDataset(spec["images_train"], spec["train_csv"], name=name)
            rows.append(self.extract(train_ds, dataset_name=name))
        return pd.DataFrame(rows)