from dataset.samplers import InstancesPerClassDataset
from dataset.config import DATASETS
from automl.search import Search
from automl.evaluator import Evaluator
from automl.results import Results
import time
from automl.config import SEARCH_SPACE
from automl.cache import EmbeddingCache
from automl.config import SUCCESSIVE_HALVING_FIDELITIES, SUCCESSIVE_HALVING_REDUCTION


class AutoML:

    def __init__(
        self,
        dataset: str,
        fidelity: int = 29,
        seed: int = 42,
        tabpfn_mode: str = "local",
    ):
        self.DATASETS = DATASETS
        if dataset not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Available datasets: {list(self.DATASETS.keys())}"
            )

        self.dataset_name = dataset
        self.dataset_cls = self.DATASETS[dataset]

        self.fidelity = fidelity
        self.seed = seed

        self.search = Search(
            SEARCH_SPACE,
            fidelity=fidelity,
            dataset_cls=self.dataset_cls,
        )
        self.tabpfn_mode = tabpfn_mode

        self.cache = EmbeddingCache()

        self.evaluator = Evaluator()
        self.results = Results(
            dataset_name=self.dataset_name,
            tabpfn_mode=self.tabpfn_mode,
            seed=self.seed,
        )

    def _load_datasets(self):

        train_dataset = self.dataset_cls(
            split="train",
            download=True,
        )

        if self.fidelity != -1:
            train_dataset = InstancesPerClassDataset(
                train_dataset,
                instances_per_class=self.fidelity,
                seed=self.seed,
            )

        test_dataset = self.dataset_cls(
            split="test",
            download=True,
        )

        print(f"Dataset : {self.dataset_name}")
        print(f"Train   : {len(train_dataset)} samples")
        print(f"Test    : {len(test_dataset)} samples")
        print(f"Classes : {self.dataset_cls.num_classes}")

        return train_dataset, test_dataset

    def successive_halving(self):

        self._load_datasets()

        configs = list(self.search)

        try:

            for fidelity in SUCCESSIVE_HALVING_FIDELITIES:

                print("\n" + "=" * 80)
                print("Successive Halving Stage")
                print("=" * 80)

                print(f"Fidelity               : {fidelity}")

                print(f"Incoming configurations: {len(configs)}")

                print("=" * 80)

                stage_results = []

                for idx, config in enumerate(
                    configs,
                    start=1,
                ):

                    print("\n" + "-" * 70)
                    print(f"Running configuration {idx}/{len(configs)}")

                    print(config)

                    print("-" * 70)

                    start = time.time()

                    score = self.evaluator.evaluate(
                        self.dataset_cls,
                        config,
                        fidelity=fidelity,
                        seed=self.seed,
                        tabpfn_mode=self.tabpfn_mode,
                    )

                    compute_time = time.time() - start

                    train_samples = fidelity * self.dataset_cls.num_classes

                    if config["augmentation"] == "randaugment":
                        train_samples *= 2

                    test_samples = len(
                        self.dataset_cls(
                            split="test",
                            download=False,
                        )
                    )

                    self.results.add(
                        fidelity=fidelity,
                        config=config,
                        score=score,
                        train_samples=train_samples,
                        test_samples=test_samples,
                        compute_time=compute_time,
                    )

                    stage_results.append(
                        (
                            score,
                            config,
                        )
                    )

                    print(f"Finished configuration {idx}")

                    print(f"Accuracy : {score:.4f}")

                    print(f"Time     : {compute_time:.2f}s")

                # -----------------------------
                # Rank configurations
                # -----------------------------

                stage_results.sort(
                    key=lambda x: x[0],
                    reverse=True,
                )

                print("\n" + "=" * 80)
                print("Stage Ranking")
                print("=" * 80)

                for rank, (score, config) in enumerate(
                    stage_results,
                    start=1,
                ):

                    print(f"{rank}. " f"Score={score:.4f}")

                    print(config)

                    print()

                # -----------------------------
                # Successive Halving
                # -----------------------------

                keep = max(
                    1,
                    len(stage_results) // SUCCESSIVE_HALVING_REDUCTION,
                )

                survivors = stage_results[:keep]

                configs = [config for _, config in survivors]

                print("\n" + "=" * 80)
                print("Successive Halving Selection")
                print("=" * 80)

                print(f"Evaluated configurations : {len(stage_results)}")

                print(f"Keeping configurations  : {keep}")

                print("\nSurvivors:")

                for rank, (score, config) in enumerate(
                    survivors,
                    start=1,
                ):

                    print("\n" + "-" * 50)

                    print(f"Rank {rank}")

                    print(f"Score: {score:.4f}")

                    print("Config:")

                    print(config)

                print("\n" + "=" * 80)

                self.results.save_stage(
                    fidelity=fidelity,
                )

                print("\nStage completed")

                if len(configs) == 1:

                    print("Only one configuration remains.")

                    break

        except Exception as e:

            print("\nSearch failed:")

            print(e)

            raise

        finally:

            print("\nCleaning embedding cache...")

            self.evaluator.cache.clear()

        # -----------------------------
        # Final best config
        # -----------------------------

        best_score, best_config = self.results.best()

        self.results.save_best(
            best_score,
            best_config,
        )

        print("\n" + "=" * 80)
        print("Search Finished")
        print("=" * 80)

        print(f"Best Score : {best_score:.4f}")

        print("Best Config:")

        print(best_config)

        return best_score, best_config

    def distill_and_train(self):
        # best_score, best_config = self.search_config()
        pass
