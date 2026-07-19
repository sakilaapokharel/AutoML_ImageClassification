from pathlib import Path
import numpy as np
import shutil


class EmbeddingCache:

    def __init__(
        self,
        root="cache",
    ):
        self.root = Path(root)

    def _get_path(
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
            / f"fidelity_{fidelity}"
            / encoder
            / f"{transform}_seed_{seed}"
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    def exists(
        self,
        dataset,
        fidelity,
        encoder,
        transform,
        seed,
    ):

        path = self._get_path(
            dataset,
            fidelity,
            encoder,
            transform,
            seed,
        )

        return (
            (path / "X_train.npy").exists()
            and (path / "y_train.npy").exists()
            and (path / "X_test.npy").exists()
            and (path / "y_test.npy").exists()
        )

    def save(
        self,
        dataset,
        fidelity,
        encoder,
        transform,
        seed,
        X_train,
        y_train,
        X_test,
        y_test,
    ):

        path = self._get_path(
            dataset,
            fidelity,
            encoder,
            transform,
            seed,
        )

        np.save(
            path / "X_train.npy",
            X_train,
        )

        np.save(
            path / "y_train.npy",
            y_train,
        )

        np.save(
            path / "X_test.npy",
            X_test,
        )

        np.save(
            path / "y_test.npy",
            y_test,
        )

        print(f"Saved embeddings to {path}")

    def clear(self):

        if self.root.exists():

            shutil.rmtree(self.root)

            print(f"Removed cache: {self.root}")

    def load(
        self,
        dataset,
        fidelity,
        encoder,
        transform,
        seed,
    ):

        path = self._get_path(
            dataset,
            fidelity,
            encoder,
            transform,
            seed,
        )

        print(f"Loading embeddings from {path}")

        X_train = np.load(path / "X_train.npy")

        y_train = np.load(path / "y_train.npy")

        X_test = np.load(path / "X_test.npy")

        y_test = np.load(path / "y_test.npy")

        return (
            X_train,
            y_train,
            X_test,
            y_test,
        )
