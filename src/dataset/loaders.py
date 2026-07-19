from torch.utils.data import DataLoader

from .datasets import FlowersDataset, EmotionsDataset
from .samplers import InstancesPerClassDataset


def make_loader(
    dataset_cls,
    split="train",
    transform=None,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    instances_per_class=None,
    **dataset_kwargs,
):
    dataset = dataset_cls(
        split=split,
        transform=transform,
        **dataset_kwargs,
    )

    if instances_per_class is not None:
        dataset = InstancesPerClassDataset(
            dataset,
            instances_per_class=instances_per_class,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
