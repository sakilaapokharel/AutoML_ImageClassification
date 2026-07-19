from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from .samplers import InstancesPerClassDataset


def load_train_test(
    dataset_cls,
    fidelity=29,
    seed=42,
    transform=None,
    **dataset_kwargs,
):

    train_dataset = dataset_cls(
        split="train",
        transform=transform,
        **dataset_kwargs,
    )

    # Apply fidelity
    if fidelity != -1:

        train_dataset = InstancesPerClassDataset(
            train_dataset,
            instances_per_class=fidelity,
            seed=seed,
        )

    test_dataset = dataset_cls(
        split="test",
        transform=transform,
        **dataset_kwargs,
    )

    # Test labels unavailable
    if not _has_labels(test_dataset):

        train_dataset, test_dataset = _split_validation(
            train_dataset,
            seed=seed,
        )

    return train_dataset, test_dataset


def _has_labels(dataset):

    labels = getattr(dataset, "_labels", None)

    return labels is not None and all(label is not None for label in labels)


def _split_validation(
    dataset,
    val_fraction=0.2,
    seed=42,
):

    labels = [dataset[i][1] for i in range(len(dataset))]

    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
    )
