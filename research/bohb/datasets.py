import os
import pandas as pd

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torch

from torch.utils.data import random_split


class CSVDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):

        self.data = pd.read_csv(csv_file)

        self.img_dir = img_dir

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        # change these names if your csv differs
        img_name = row["image_file_name"]

        label = row["label"]

        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path)

        return image, int(label)


def load_dataset(dataset_name, data_root, batch_size):

    train_csv = os.path.join(data_root, dataset_name, "train.csv")

    test_csv = os.path.join(data_root, dataset_name, "test.csv")

    train_img_dir = os.path.join(data_root, dataset_name, "images_train")

    test_img_dir = os.path.join(data_root, dataset_name, "images_test")

    # Raw datasets (no transform yet)
    full_train_ds = CSVDataset(train_csv, train_img_dir, transform=None)

    test_ds = CSVDataset(test_csv, test_img_dir, transform=None)

    # 80/20 split
    train_size = int(0.8 * len(full_train_ds))

    val_size = len(full_train_ds) - train_size

    train_ds, val_ds = random_split(
        full_train_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    num_classes = pd.read_csv(train_csv)["label"].nunique()

    # detect image properties here
    sample_img, _ = full_train_ds[0]

    img_size = sample_img.size

    if sample_img.mode == "L":
        in_channels = 1
    else:
        in_channels = 3

    return (train_ds, val_ds, test_ds, num_classes, img_size, in_channels)


class TransformedDataset(torch.utils.data.Dataset):

    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):

        img, label = self.base[idx]

        img = self.transform(img)

        return img, label
