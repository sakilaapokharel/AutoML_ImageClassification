from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass


import torch
import numpy as np


class TorchEmbedder(BaseEmbedder):

    def __init__(
        self,
        encoder,
        device="cuda",
    ):
        self.encoder = encoder
        self.device = device

        self.encoder.to(device)
        self.encoder.eval()

    def fit(self, X, y=None):
        """
        Frozen embedder.
        No training required.
        """
        return self

    @torch.no_grad()
    def predict(self, loader):

        embeddings = []

        for x, _ in loader:
            x = x.to(self.device)

            z = self.encoder(x)

            embeddings.append(z.cpu().numpy())

        return np.concatenate(embeddings)

    def predict_proba(self, loader):
        """
        Optional.
        Usually done by classifier head,
        not encoder.
        """
        raise NotImplementedError
