from pathlib import Path
import numpy as np
import shutil


class EmbeddingCache:

    def __init__(self, root="cache"):
        self.root = Path(root)

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------

    def _train_path(
        self,
        dataset,
        fidelity,
        encoder,
        transform,
        seed,
    ):

        path = (
            self.root
            / dataset
            / "train"
            / f"fidelity_{fidelity}"
            / encoder
            / f"{transform}_seed_{seed}"
        )

        path.mkdir(parents=True, exist_ok=True)

        return path

    def _val_path(
        self,
        dataset,
        fidelity,
        encoder,
        transform,
        seed,
    ):

        path = (
            self.root
            / dataset
            / "val"
            / f"fidelity_{fidelity}"
            / encoder
            / f"{transform}_seed_{seed}"
        )

        path.mkdir(parents=True, exist_ok=True)

        return path

    # --------------------------------------------------
    # Existence
    # --------------------------------------------------

    def train_exists(
        self,
        dataset,
        fidelity,
        encoder,
        transform,
        seed,
    ):

        path = self._train_path(
            dataset,
            fidelity,
            encoder,
            transform,
            seed,
        )

        return (path / "X_train.npy").exists() and (path / "y_train.npy").exists()

    def val_exists(
        self,
        dataset,
        fidelity,
        encoder,
        transform,
        seed,
    ):

        path = self._val_path(
            dataset,
            fidelity,
            encoder,
            transform,
            seed,
        )

        return (path / "X_val.npy").exists() and (path / "y_val.npy").exists()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    def save_train(
        self,
        dataset,
        fidelity,
        encoder,
        transform,
        seed,
        X_train,
        y_train,
    ):

        path = self._train_path(
            dataset,
            fidelity,
            encoder,
            transform,
            seed,
        )

        np.save(path / "X_train.npy", X_train)
        np.save(path / "y_train.npy", y_train)

        print(f"Saved train embeddings: {path}")

    def save_val(
        self,
        dataset,
        fidelity,
        encoder,
        transform,
        seed,
        X_val,
        y_val,
    ):

        path = self._val_path(
            dataset,
            fidelity,
            encoder,
            transform,
            seed,
        )

        np.save(path / "X_val.npy", X_val)
        np.save(path / "y_val.npy", y_val)

        print(f"Saved val embeddings: {path}")

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    def load_train(
        self,
        dataset,
        fidelity,
        encoder,
        transform,
        seed,
    ):

        path = self._train_path(
            dataset,
            fidelity,
            encoder,
            transform,
            seed,
        )

        print(f"Loading train embeddings: {path}")

        return (
            np.load(path / "X_train.npy"),
            np.load(path / "y_train.npy"),
        )

    def load_val(
        self,
        dataset,
        fidelity,
        encoder,
        transform,
        seed,
    ):

        path = self._val_path(
            dataset,
            fidelity,
            encoder,
            transform,
            seed,
        )

        print(f"Loading val embeddings: {path}")

        return (
            np.load(path / "X_val.npy"),
            np.load(path / "y_val.npy"),
        )

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    def clear(self):

        if self.root.exists():

            shutil.rmtree(self.root)

            print(f"Removed cache: {self.root}")
