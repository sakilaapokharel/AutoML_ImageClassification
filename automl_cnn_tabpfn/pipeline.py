from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from data_utils import ImageCSVDataset, load_csv, sample_per_class
from feature_extractor import build_backbone, build_transforms, extract_features, get_device

import os
from dotenv import load_dotenv
load_dotenv()

def extract_dataset_features(
    dataset_dir,
    model_name,
    n_per_class,
    seed=42,
    batch_size=32,
    resize_size=224,
    augment=False,
):
    dataset_dir = Path(dataset_dir)
    train_df, file_col, label_col = load_csv(dataset_dir / "train.csv")
    test_df, test_file_col, test_label_col = load_csv(dataset_dir / "test.csv")

    sampled_train_df = sample_per_class(train_df, label_col, n_per_class, seed=seed)

    model, embed_dim = build_backbone(model_name)
    device = get_device()

    train_transform, eval_transform, randaug_magnitude = build_transforms(resize_size=resize_size, augment=augment)

    if resize_size == 0:
        batch_size = 1

    test_ds = ImageCSVDataset(test_df, test_file_col, test_label_col, dataset_dir / "images_test", eval_transform)

    if augment:
        clean_ds = ImageCSVDataset(sampled_train_df, file_col, label_col, dataset_dir / "images_train", eval_transform)
        clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False)
        X_clean, y_clean = extract_features(model, clean_loader, device)

        aug_ds = ImageCSVDataset(sampled_train_df, file_col, label_col, dataset_dir / "images_train", train_transform)
        aug_loader = DataLoader(aug_ds, batch_size=batch_size, shuffle=False)
        X_aug, y_aug = extract_features(model, aug_loader, device)

        X_train = np.concatenate([X_clean, X_aug], axis=0)
        y_train = np.concatenate([y_clean, y_aug], axis=0)
    else:
        train_ds = ImageCSVDataset(sampled_train_df, file_col, label_col, dataset_dir / "images_train", train_transform)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        X_train, y_train = extract_features(model, train_loader, device)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    X_test, y_test = extract_features(model, test_loader, device)

    return X_train, y_train, X_test, y_test, embed_dim, randaug_magnitude


def run(
    dataset_dir,
    model_name,
    n_per_class,
    pca_dim=100,
    seed=42,
    batch_size=32,
    resize_size=224,
    augment=False,
):
    X_train, y_train, X_test, y_test, embed_dim, randaug_magnitude = extract_dataset_features(
        dataset_dir, model_name, n_per_class, seed, batch_size, resize_size, augment
    )

    if pca_dim:
        max_components = min(X_train.shape[0], X_train.shape[1])
        effective_dim = min(pca_dim, max_components)
        if effective_dim < X_train.shape[1]:
            pca = PCA(n_components=effective_dim, random_state=seed)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

    from tabpfn_client import TabPFNClassifier
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_df = load_csv(Path(dataset_dir) / "test.csv")[0]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "n_train": len(X_train),
        "n_test": len(test_df),
        "embed_dim": X_train.shape[1],
        "model": model_name,
        "n_per_class": n_per_class,
        "augment": augment,
        "randaug_magnitude": randaug_magnitude,
        "resize_size": resize_size,
    }