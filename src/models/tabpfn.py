import torch
import numpy as np


class TabPFNModel:

    def __init__(
        self,
        mode="local",
        device="cuda",
        seed=42,
        n_estimators=1,
    ):

        self.mode = mode
        self.seed = seed
        self.n_estimators = n_estimators

        if mode == "local":

            from tabpfn import TabPFNClassifier

            self.model = TabPFNClassifier(
                device=device,
                n_estimators=n_estimators,
                random_state=seed,
            )

        elif mode == "client":

            from tabpfn_client import TabPFNClassifier

            self.model = TabPFNClassifier(
                n_estimators=n_estimators,
                random_state=seed,
            )

        else:

            raise ValueError(
                f"Unknown TabPFN mode '{mode}'. " "Choose 'local' or 'client'."
            )

    def fit(
        self,
        X_train,
        y_train,
    ):

        self.model.fit(
            X_train,
            y_train,
        )

        return self

    def predict(
        self,
        X_test,
        batch_size=100,
    ):

        if len(X_test) <= batch_size:
            return self.model.predict(X_test)

        print(
            f"TabPFN predict split into "
            f"{(len(X_test)+batch_size-1)//batch_size} batches"
        )

        predictions = []

        for start in range(0, len(X_test), batch_size):

            end = min(start + batch_size, len(X_test))

            print(f"Predicting {start}:{end}")

            pred = self.model.predict(
                X_test[start:end]
            )

            predictions.append(pred)

        return np.concatenate(predictions)
    def predict_proba(
        self,
        X_test,
        batch_size=100,
    ):

        if len(X_test) <= batch_size:
            return self.model.predict_proba(X_test)

        print(
            f"TabPFN predict_proba split into "
            f"{(len(X_test)+batch_size-1)//batch_size} batches"
        )

        probabilities = []

        for start in range(0, len(X_test), batch_size):

            end = min(start + batch_size, len(X_test))

            print(f"Predicting probabilities {start}:{end}")

            probs = self.model.predict_proba(
                X_test[start:end]
            )

            probabilities.append(probs)

        return np.vstack(probabilities)