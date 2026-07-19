from sklearn.decomposition import PCA
import numpy as np


class PCAReducer:

    def __init__(
        self,
        dim,
        seed=42,
    ):
        self.dim = dim
        self.seed = seed
        self.reducer = None

    def fit_transform(
        self,
        X,
    ):

        if self.dim is None:
            return X

        self.reducer = PCA(
            n_components=self.dim,
            random_state=self.seed,
        )

        return self.reducer.fit_transform(X)

    def transform(
        self,
        X,
    ):

        if self.reducer is None:
            return X

        return self.reducer.transform(X)
