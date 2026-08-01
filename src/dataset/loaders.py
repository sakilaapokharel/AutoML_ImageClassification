from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from src.dataset.samplers import InstancesPerClassDataset
from src.dataset.transforms import build_transform, TransformDataset


def load_datasets(
    dataset_cls,
    preprocess_policy,
    resize=224,
    fidelity=-1,
    seed=42,
):

    train_transform = build_transform(
        image_size=resize,
        preprocess_policy=preprocess_policy,
        train=True,
    )

    val_transform = build_transform(
        image_size=resize,
        preprocess_policy=preprocess_policy,
        train=False,
    )

    print("Train Transform")
    print(train_transform)

    print("Val Transform")
    print(val_transform)

    dataset = dataset_cls(
        split="train",
        transform=None,
        download=True,
    )

    labels = dataset._labels

    # -------------------------------------------------
    # Use the entire training set (no validation split)
    # -------------------------------------------------
    if fidelity == -1:

        train_dataset = TransformDataset(
            dataset,
            train_transform,
        )

        print(f"\nDataset: {dataset_cls._dataset_name}")
        print(f"Train: {len(train_dataset)}")

        sample_image, sample_label = train_dataset[0]

        print("\nImage information")
        print(f"Image shape: {tuple(sample_image.shape)}")
        print(f"Image dtype: {sample_image.dtype}")

        train_stats = Counter(labels)

        print("\nClass distribution")
        print(f"{'Class':<10}{'Train':<10}")

        for cls in sorted(train_stats):
            print(f"{cls:<10}{train_stats[cls]:<10}")

        return train_dataset

    # -------------------------------------------------
    # Train/validation split
    # -------------------------------------------------
    indices = list(range(len(dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )

    train_subset = InstancesPerClassDataset(
        dataset,
        indices=train_idx,
        instances_per_class=fidelity,
        seed=seed,
    )

    val_subset = Subset(
        dataset,
        val_idx,
    )

    train_dataset = TransformDataset(
        train_subset,
        train_transform,
    )

    val_dataset = TransformDataset(
        val_subset,
        val_transform,
    )

    print(f"\nDataset: {dataset_cls._dataset_name}")
    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")

    sample_image, sample_label = train_dataset[0]

    print("\nImage information")
    print(f"Image shape: {tuple(sample_image.shape)}")
    print(f"Image dtype: {sample_image.dtype}")

    train_labels_final = [labels[idx] for idx in train_subset.selected_indices]
    val_labels_final = [labels[idx] for idx in val_idx]

    train_stats = Counter(train_labels_final)
    val_stats = Counter(val_labels_final)

    print("\nClass distribution")
    print(f"{'Class':<10}{'Train':<10}{'Val':<10}")

    for cls in sorted(set(labels)):
        print(f"{cls:<10}" f"{train_stats[cls]:<10}" f"{val_stats[cls]:<10}")

    return train_dataset, val_dataset


def load_test_datasets(dataset_cls, seed=42):
    test_transform = build_transform(
        train=False,
    )
    dataset = dataset_cls(
        split="test",
        transform=None,
        download=True,
    )
    test_dataset = TransformDataset(
        dataset,
        test_transform,
    )
    return test_dataset
