import torch
from torch.utils.data import WeightedRandomSampler


def get_sampler(mode, train_targets, num_classes):

    if mode == "random" or mode is None:
        return None, True  # shuffle = True

    if mode == "weighted":
        counts = torch.bincount(
            torch.tensor(train_targets), minlength=num_classes
        ).float()

        weights = 1.0 / counts.clamp(min=1)
        sample_weights = weights[torch.tensor(train_targets)]

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        return sampler, False  # shuffle must be False

    raise ValueError(f"Unknown sampler mode: {mode}")
