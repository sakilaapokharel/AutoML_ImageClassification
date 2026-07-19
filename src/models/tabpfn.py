from __future__ import annotations

import numpy as np


def get_tabpfn_classifier(
    mode: str = "local",
    device: str = "cuda",
    seed: int = 42,
    n_estimators: int = 1,
):
    """
    Create a TabPFN classifier.

    Parameters
    ----------
    mode:
        "local" or "client"

    device:
        Device for local TabPFN inference.

    seed:
        Random seed.

    n_estimators:
        Number of TabPFN ensemble estimators.
    """

    if mode == "client":
        from tabpfn_client import TabPFNClassifier

        model = TabPFNClassifier(
            random_state=seed,
            n_estimators=n_estimators,
        )

    elif mode == "local":
        from tabpfn import TabPFNClassifier

        model = TabPFNClassifier(
            device=device,
            random_state=seed,
            n_estimators=n_estimators,
        )

    else:
        raise ValueError(
            f"Unknown TabPFN mode '{mode}'. " "Choose 'local' or 'client'."
        )

    return model


class TabPFNModel:

    def __init__(
        self,
        mode: str = "local",
        device: str = "cuda",
        seed: int = 42,
        n_estimators: int = 1,
    ):
        self.mode = mode
        self.seed = seed
        self.n_estimators = n_estimators

        self.model = get_tabpfn_classifier(
            mode=mode,
            device=device,
            seed=seed,
            n_estimators=n_estimators,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        self.model.fit(
            X_train,
            y_train,
        )

        return self

    def predict(
        self,
        X_test: np.ndarray,
    ):
        return self.model.predict(X_test)

    def predict_proba(
        self,
        X_test: np.ndarray,
    ):
        return self.model.predict_proba(X_test)
