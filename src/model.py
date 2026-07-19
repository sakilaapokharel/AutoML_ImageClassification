from dataset.samplers import InstancesPerClassDataset
from dataset.config import DATASETS
from automl.search import Search
from automl.evaluator import Evaluator
from automl.results import Results
import time
from automl.config import SEARCH_SPACE


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

        self.search = Search(config_space=SEARCH_SPACE)
        self.tabpfn_mode = tabpfn_mode

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

    def search_config(self):

        train_dataset, test_dataset = self._load_datasets()

        total = self.search.size()

        for idx, config in enumerate(
            self.search,
            start=1,
        ):

            print("\n" + "=" * 60)
            print(f"Configuration {idx}/{total}")
            print(config)
            print("=" * 60)

            start = time.time()

            score = self.evaluator.evaluate(
                self.dataset_cls,
                config,
                self.fidelity,
                seed=self.seed,
                tabpfn_mode=self.tabpfn_mode,
            )

            compute_time = time.time() - start

            if config["augmentation"] == "randaugment":
                train_samples = len(train_dataset) * 2
            else:
                train_samples = len(train_dataset)
                test_samples = len(test_dataset)

            self.results.add(
                config=config,
                score=score,
                train_samples=train_samples,
                test_samples=test_samples,
                compute_time=compute_time,
            )

        return self
