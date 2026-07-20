from pathlib import Path
import numpy as np
import shutil


class EmbeddingCache:

    def __init__(
        self,
        root="cache",
    ):
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

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    def _test_path(
        self,
        dataset,
        encoder,
        transform,
        seed,
    ):

        path = self.root / dataset / "test" / encoder / f"{transform}_seed_{seed}"

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

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

    def test_exists(
        self,
        dataset,
        encoder,
        transform,
        seed,
    ):

        path = self._test_path(
            dataset,
            encoder,
            transform,
            seed,
        )

        return (path / "X_test.npy").exists() and (path / "y_test.npy").exists()

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

        np.save(
            path / "X_train.npy",
            X_train,
        )

        np.save(
            path / "y_train.npy",
            y_train,
        )

        print(f"Saved train embeddings: {path}")

    def save_test(
        self,
        dataset,
        encoder,
        transform,
        seed,
        X_test,
        y_test,
    ):

        path = self._test_path(
            dataset,
            encoder,
            transform,
            seed,
        )

        np.save(
            path / "X_test.npy",
            X_test,
        )

        np.save(
            path / "y_test.npy",
            y_test,
        )

        print(f"Saved test embeddings: {path}")

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

    def load_test(
        self,
        dataset,
        encoder,
        transform,
        seed,
    ):

        path = self._test_path(
            dataset,
            encoder,
            transform,
            seed,
        )

        print(f"Loading test embeddings: {path}")

        return (
            np.load(path / "X_test.npy"),
            np.load(path / "y_test.npy"),
        )

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    def clear(self):

        if self.root.exists():

            shutil.rmtree(self.root)

            print(f"Removed cache: {self.root}")
