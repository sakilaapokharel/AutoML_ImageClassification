import torch
import numpy as np


class TabPFNModel:

    # tabpfn_client enforces a max of 20,000,000 *cells* (rows x columns) per
    # call, not a max row count. A fixed row-count batch size (e.g. 2,000,000)
    # only respects this limit when num_features is small -- with more
    # features, rows x features can exceed 20M even after "batching". Compute
    # the safe row count per batch dynamically instead.
    MAX_CLIENT_PREDICT_CELLS = 20_000_000
    MAX_CLIENT_TRAIN_CELLS = 100_000_000

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

    def _client_batch_size(self, X):

        num_features = X.shape[1] if X.ndim > 1 else 1

        # small safety margin below the hard limit, in case the server
        # counts a few extra cells (e.g. an implicit index/label column)
        safe_cells = int(self.MAX_CLIENT_PREDICT_CELLS * 0.95)

        return max(1, safe_cells // num_features)

    def fit(self, X_train, y_train):

        if self.mode == "client":

            num_features = X_train.shape[1]

            max_rows = self.MAX_CLIENT_TRAIN_CELLS // num_features

            if len(X_train) > max_rows:

                rng = np.random.default_rng(self.seed)

                indices = rng.choice(
                    len(X_train),
                    size=max_rows,
                    replace=False,
                )

                X_train = X_train[indices]
                y_train = y_train[indices]

                print(
                    f"Randomly selected {max_rows:,} training samples "
                    f"using seed={self.seed}"
                )

            print(
                f"Training TabPFN with "
                f"{len(X_train):,} samples × "
                f"{num_features:,} features = "
                f"{len(X_train) * num_features:,} cells"
            )

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
        if self.mode == "client":
            batch_size = self._client_batch_size(X_test)

        if len(X_test) <= batch_size:
            return self.model.predict(X_test)

        print(
            f"TabPFN predict split into "
            f"{(len(X_test)+batch_size-1)//batch_size} batches "
            f"(batch_size={batch_size})"
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
        if self.mode == "client":
            batch_size = self._client_batch_size(X_test)

        if len(X_test) <= batch_size:
            return self.model.predict_proba(X_test)

        print(
            f"TabPFN predict_proba split into "
            f"{(len(X_test)+batch_size-1)//batch_size} batches "
            f"(batch_size={batch_size})"
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