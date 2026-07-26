import random
from collections import defaultdict
from torch.utils.data import Dataset


class InstancesPerClassDataset:

    def __init__(
        self,
        dataset,
        indices,
        instances_per_class,
        seed=42,
    ):

        self.dataset = dataset

        random.seed(seed)

        self.selected_indices = []

        class_indices = defaultdict(list)

        for idx in indices:

            label = dataset._labels[idx]

            class_indices[label].append(idx)

        for label, idxs in class_indices.items():

            selected = random.sample(
                idxs,
                min(
                    instances_per_class,
                    len(idxs),
                ),
            )

            self.selected_indices.extend(selected)

    def __len__(self):

        return len(self.selected_indices)

    def __getitem__(self, index):

        real_idx = self.selected_indices[index]

        return self.dataset[real_idx]
