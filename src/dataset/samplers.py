import random
from collections import defaultdict
from torch.utils.data import Dataset


class InstancesPerClassDataset(Dataset):

    def __init__(
        self,
        dataset,
        instances_per_class,
        seed=42,
    ):
        self.dataset = dataset

        rng = random.Random(seed)

        class_indices = defaultdict(list)

        for idx, label in enumerate(dataset._labels):
            class_indices[label].append(idx)

        self.indices = []

        for indices in class_indices.values():

            rng.shuffle(indices)

            max_instances = min(
                instances_per_class,
                len(indices),
            )

            self.indices.extend(indices[:max_instances])

        rng.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
