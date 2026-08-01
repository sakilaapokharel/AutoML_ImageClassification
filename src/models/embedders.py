import numpy as np
import torch
from torch.utils.data import DataLoader

from src.automl.utils import print_progress


class Embedder:

    def __init__(
        self,
        encoder,
        device="cuda",
        batch_size=64,
        num_workers=0,
        pin_memory=True,
    ):

        self.encoder = encoder
        self.device = device

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def fit(self):
        """
        Frozen encoder.
        No training required.
        """
        return self

    @torch.no_grad()
    def embed(
        self,
        dataset,
    ):

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        embeddings = []
        labels = []

        total = len(dataset)
        seen = 0

        self.fit()

        for x, y in loader:

            x = x.to(
                self.device,
                non_blocking=True,
            )

            z = self.encoder(x)

            embeddings.append(z.cpu().numpy())

            labels.append(y.numpy())

            seen += len(x)

            print_progress(
                seen,
                total,
                every=500,
            )

        return (
            np.concatenate(
                embeddings,
                axis=0,
            ),
            np.concatenate(
                labels,
                axis=0,
            ),
        )
