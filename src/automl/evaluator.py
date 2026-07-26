import numpy as np
from models.encoders import get_encoder
from models.reducers import PCAReducer
from models.embedders import Embedder
from automl.cache import EmbeddingCache
from dataset.loaders import load_datasets
from automl.utils import print_header, print_step


class Evaluator:

    def __init__(self, tabpfn_model, device):

        self.cache = EmbeddingCache()

        self.tabpfn_model = tabpfn_model
        self.device = device

    def evaluate(
        self,
        dataset_cls,
        config,
        fidelity,
        seed=42,
    ):
        print_header(f"Encoder: {config['encoder']}")

        train_dataset, val_dataset = load_datasets(
            dataset_cls=dataset_cls,
            preprocess_policy=config["preprocess_policy"],
            resize=224,
            fidelity=fidelity,
            seed=seed,
        )

        print(f"Train samples: {len(train_dataset)}")

        print(f"Val samples: {len(val_dataset)}")

        dataset_name = dataset_cls._dataset_name

        train_transform_name = f"{config['preprocess_policy']}"

        val_transform_name = f"{config['preprocess_policy']}"

        encoder, embedding = get_encoder(config["encoder"])
        encoder.eval()
        encoder.to(self.device)

        embedder = Embedder(encoder, device=self.device)

        # ==================================================
        # TRAIN EMBEDDINGS
        # ==================================================

        if self.cache.train_exists(
            dataset_name,
            fidelity,
            config["encoder"],
            train_transform_name,
            seed,
        ):

            print_step("Loading cached train embeddings")

            X_train, y_train = self.cache.load_train(
                dataset_name,
                fidelity,
                config["encoder"],
                train_transform_name,
                seed,
            )

        else:

            print_step("Embedding train data")

            X_train, y_train = embedder.embed(train_dataset)

            self.cache.save_train(
                dataset_name,
                fidelity,
                config["encoder"],
                train_transform_name,
                seed,
                X_train,
                y_train,
            )

        # ==================================================
        # Val EMBEDDINGS
        # ==================================================

        if self.cache.val_exists(
            dataset_name,
            fidelity,
            config["encoder"],
            val_transform_name,
            seed,
        ):

            print_step("Loading cached val embeddings")

            X_val, y_val = self.cache.load_val(
                dataset_name,
                fidelity,
                config["encoder"],
                val_transform_name,
                seed,
            )

        else:

            print_step("Embedding val data")

            X_val, y_val = embedder.embed(val_dataset)

            self.cache.save_val(
                dataset_name,
                fidelity,
                config["encoder"],
                val_transform_name,
                seed,
                X_val,
                y_val,
            )

        # ==================================================
        # PCA
        # ==================================================

        print_step("Reducing dimensions")

        reducer = PCAReducer(
            config["embedding_dim"],
            seed,
        )

        X_train = reducer.fit_transform(X_train)

        X_val = reducer.transform(X_val)

        # ==================================================
        # TabPFN
        # ==================================================

        print_step("Training TabPFN")

        self.tabpfn_model.fit(
            X_train,
            y_train,
        )

        print_step("Evaluating")

        predictions = self.tabpfn_model.predict(X_val)

        score = np.mean(predictions == y_val)

        print(f"\nScore: {score:.4f}")

        return score
