import torch


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
    ):

        return self.model.predict(X_test)

    def predict_proba(
        self,
        X_test,
    ):

        return self.model.predict_proba(X_test)
