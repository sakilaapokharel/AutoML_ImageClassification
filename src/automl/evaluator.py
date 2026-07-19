import torch
import numpy as np

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from dataset.samplers import InstancesPerClassDataset

from models.encoders import get_encoder
from models.tabpfn import TabPFNModel

from automl.utils import (
    print_header,
    print_step,
    print_progress,
)
from models.reducers import PCAReducer


class Evaluator:

    def _get_transform(
        self,
        config,
    ):

        base = [transforms.Resize((config["resize"], config["resize"]))]

        if config["augmentation"] == "randaugment":

            train_transform = transforms.Compose(
                base
                + [
                    transforms.RandAugment(),
                    transforms.ToTensor(),
                ]
            )

            original_transform = transforms.Compose(
                base
                + [
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

            original_transform = train_transform

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

        train_dataset = dataset_cls(
            split="train",
            transform=original_transform,
            download=False,
        )

        if fidelity != -1:

            train_dataset = InstancesPerClassDataset(
                train_dataset,
                instances_per_class=fidelity,
                seed=seed,
            )

        # Add augmented copy
        if config["augmentation"] == "randaugment":

            aug_dataset = dataset_cls(
                split="train",
                transform=train_transform,
                download=False,
            )

            if fidelity != -1:

                aug_dataset = InstancesPerClassDataset(
                    aug_dataset,
                    instances_per_class=fidelity,
                    seed=seed,
                )

            train_dataset = ConcatDataset(
                [
                    train_dataset,
                    aug_dataset,
                ]
            )

        test_dataset = dataset_cls(
            split="test",
            transform=test_transform,
            download=False,
        )

        return train_dataset, test_dataset

    def _embed(
        self,
        encoder,
        dataset,
    ):

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
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

    def evaluate(self, dataset_cls, config, fidelity, seed=42, tabpfn_mode="local"):

        print_header(f"Encoder: {config['encoder']}")

        print_step("Loading datasets")

        train_dataset, test_dataset = self._load_datasets(
            dataset_cls,
            config,
            fidelity,
            seed,
        )

        print(f"Train samples: {len(train_dataset)}")

        print(f"Test samples: {len(test_dataset)}")

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

        print_step("Reducing Dimensions")

        reducer = PCAReducer(
            config["embedding_dim"],
            seed,
        )

        X_train = reducer.fit_transform(X_train)

        X_test = reducer.transform(X_test)

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
