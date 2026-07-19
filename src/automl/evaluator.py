import torch
import numpy as np

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from dataset.samplers import InstancesPerClassDataset
from dataset.loaders import TransformDataset

from models.encoders import get_encoder
from models.tabpfn import TabPFNModel
from models.reducers import PCAReducer

from automl.cache import EmbeddingCache

from automl.utils import (
    print_header,
    print_step,
    print_progress,
)


class Evaluator:

    def __init__(self):

        self.cache = EmbeddingCache()

    def _get_transform(
        self,
        config,
    ):

        base = []

        if config["resize"] is not None:

            base.append(
                transforms.Resize(
                    (
                        config["resize"],
                        config["resize"],
                    )
                )
            )

        if config["augmentation"] == "randaugment":

            train_transform = transforms.Compose(
                base
                + [
                    transforms.RandAugment(),
                    transforms.ToTensor(),
                ]
            )

        else:

            train_transform = transforms.Compose(
                base
                + [
                    transforms.ToTensor(),
                ]
            )

        original_transform = transforms.Compose(
            base
            + [
                transforms.ToTensor(),
            ]
        )

        test_transform = transforms.Compose(
            base
            + [
                transforms.ToTensor(),
            ]
        )

        return (
            original_transform,
            train_transform,
            test_transform,
        )

    def _load_datasets(
        self,
        dataset_cls,
        config,
        fidelity,
        seed,
    ):

        (
            original_transform,
            train_transform,
            test_transform,
        ) = self._get_transform(config)

        print_step("Loading raw training dataset")

        raw_train = dataset_cls(
            split="train",
            transform=None,
            download=False,
        )

        #
        # Fidelity selection
        #
        if fidelity != -1:

            sampled_train = InstancesPerClassDataset(
                raw_train,
                instances_per_class=fidelity,
                seed=seed,
            )

        else:

            sampled_train = raw_train

        #
        # Original images
        #
        original_dataset = TransformDataset(
            sampled_train,
            original_transform,
        )

        #
        # Original + RandAugment
        #
        if config["augmentation"] == "randaugment":

            augmented_dataset = TransformDataset(
                sampled_train,
                train_transform,
            )

            train_dataset = ConcatDataset(
                [
                    original_dataset,
                    augmented_dataset,
                ]
            )

        else:

            train_dataset = original_dataset

        #
        # Test
        #
        test_dataset = dataset_cls(
            split="test",
            transform=test_transform,
            download=False,
        )

        return (
            train_dataset,
            test_dataset,
        )

    def _embed(
        self,
        encoder,
        dataset,
    ):

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            pin_memory=True,
        )

        embeddings = []
        labels = []

        total = len(dataset)
        seen = 0

        with torch.no_grad():

            for x, y in loader:

                x = x.cuda()

                z = encoder(x)

                embeddings.append(z.cpu().numpy())

                labels.append(y.numpy())

                seen += len(x)

                print_progress(
                    seen,
                    total,
                    every=500,
                )

        return (
            np.concatenate(embeddings),
            np.concatenate(labels),
        )

    def evaluate(
        self,
        dataset_cls,
        config,
        fidelity,
        seed=42,
        tabpfn_mode="local",
    ):

        print_header(f"Encoder: {config['encoder']}")

        train_dataset, test_dataset = self._load_datasets(
            dataset_cls,
            config,
            fidelity,
            seed,
        )

        print(f"Train samples: {len(train_dataset)}")

        print(f"Test samples: {len(test_dataset)}")

        transform_name = f"resize_{config['resize']}_" f"{config['augmentation']}"

        #
        # CNN CACHE
        #
        if self.cache.exists(
            dataset_cls._dataset_name,
            fidelity,
            config["encoder"],
            transform_name,
            seed,
        ):

            print_step("Loading cached embeddings")

            (
                X_train,
                y_train,
                X_test,
                y_test,
            ) = self.cache.load(
                dataset_cls._dataset_name,
                fidelity,
                config["encoder"],
                transform_name,
                seed,
            )

        else:

            print_step("Loading encoder")

            encoder, embedding_dim = get_encoder(config["encoder"])

            encoder.eval()
            encoder.cuda()

            print_step("Embedding train data")

            X_train, y_train = self._embed(
                encoder,
                train_dataset,
            )

            print_step("Embedding test data")

            X_test, y_test = self._embed(
                encoder,
                test_dataset,
            )

            print_step("Saving embeddings")

            self.cache.save(
                dataset_cls._dataset_name,
                fidelity,
                config["encoder"],
                transform_name,
                seed,
                X_train,
                y_train,
                X_test,
                y_test,
            )

        #
        # PCA
        #
        print_step("Reducing dimensions")

        reducer = PCAReducer(
            config["embedding_dim"],
            seed,
        )

        X_train = reducer.fit_transform(X_train)

        X_test = reducer.transform(X_test)

        #
        # TabPFN
        #
        print_step("Training TabPFN")

        model = TabPFNModel(
            mode=tabpfn_mode,
            seed=seed,
            n_estimators=1,
        )

        model.fit(
            X_train,
            y_train,
        )

        print_step("Evaluating")

        predictions = model.predict(X_test)

        score = np.mean(predictions == y_test)

        print(f"\nScore: {score:.4f}")

        return score
