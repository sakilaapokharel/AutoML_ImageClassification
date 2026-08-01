import time

import optuna
import torch

from src.automl.bohb import BOHBSearch
from src.automl.cache import EmbeddingCache
from src.automl.config import (
    SEARCH_SPACE,
    SUCCESSIVE_HALVING_FIDELITIES,
    SUCCESSIVE_HALVING_REDUCTION,
)
from src.automl.evaluator import Evaluator
from src.automl.results import Results
from src.automl.search import Search
from src.automl.successive_halving import SuccessiveHalvingSearch
from src.automl.utils import estimate_bohb_trials
from src.dataset.config import DATASETS
from src.models.tabpfn import TabPFNModel

class AutoML:

    def __init__(
        self,
        dataset: str,
        seed: int = 42,
        tabpfn_mode: str = "local",
        search_strategy: str = "successive_halving",
    ):
        self.DATASETS = DATASETS
        if dataset not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Available datasets: {list(self.DATASETS.keys())}"
            )

        self.dataset_name = dataset
        self.dataset_cls = self.DATASETS[dataset]
        self.num_classes = self.dataset_cls.num_classes
        self.seed = seed
        self.tabpfn_mode = tabpfn_mode

        self.search_strategy = search_strategy
        self.total_time = 0.0
        self.best_score = 0.0
        self.evaluation_count = 0

        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tabpfn_model = TabPFNModel(
            mode=tabpfn_mode,
            seed=seed,
            device=self.device,
        )

        self.search = Search(
            SEARCH_SPACE,
        )
        self.cache = EmbeddingCache()

        self.evaluator = Evaluator(tabpfn_model=self.tabpfn_model, device=self.device)
        self.results = Results(
            dataset_name=self.dataset_name,
            tabpfn_mode=self.tabpfn_mode,
            seed=self.seed,
            search_strategy=self.search_strategy,
        )

    def successive_halving(self):

        searcher = SuccessiveHalvingSearch(
            search=self.search,
            fidelities=SUCCESSIVE_HALVING_FIDELITIES,
            reduction_factor=SUCCESSIVE_HALVING_REDUCTION,
        )

        def evaluate(config, fidelity):

            num_instance_per_class = fidelity // self.num_classes
            start = time.time()

            score = self.evaluator.evaluate(
                self.dataset_cls,
                config,
                fidelity=num_instance_per_class,
                seed=self.seed,
            )
            compute_time = time.time() - start
            self.total_time += compute_time
            self.evaluation_count += 1

            self.best_score = max(
                self.best_score,
                score,
            )

            self.results.add(
                fidelity=fidelity,
                config=config,
                score=score,
                train_samples=fidelity,
                num_instance_per_class=num_instance_per_class,
                compute_time=compute_time,
                cumulative_time=self.total_time,
                best_score_so_far=self.best_score,
                evaluation_count=self.evaluation_count,
            )

            return score

        try:

            searcher.run(
                evaluate=evaluate,
                save_stage=self.results.save_stage,
            )

        finally:

            print("\nCleaning embedding cache...")

            self.evaluator.cache.clear()

        best_score, best_config = self.results.best()

        self.results.save_best(
            best_score,
            best_config,
            cumulative_time=self.total_time,
            evaluation_count=self.evaluation_count,
        )

        print("\nSearch Finished")

        print(f"Best Score : {best_score:.4f}")

        print(best_config)

        return best_score, best_config

    def bohb(self):

        print("\n" + "=" * 80)
        print("Starting BOHB Search")
        print("=" * 80)

        n_trials = estimate_bohb_trials(search_space=SEARCH_SPACE)
        print("Number of Trials ", n_trials)

        bohb = BOHBSearch(
            search_space=SEARCH_SPACE,
            fidelities=SUCCESSIVE_HALVING_FIDELITIES,
            reduction_factor=SUCCESSIVE_HALVING_REDUCTION,
            n_trials=n_trials,
            seed=self.seed,
        )

        def objective(trial, config):

            print("\nTrial config:")
            print(config)

            final_score = None

            for fidelity in SUCCESSIVE_HALVING_FIDELITIES:

                num_instance_per_class = fidelity // self.num_classes

                start = time.time()

                score = self.evaluator.evaluate(
                    self.dataset_cls,
                    config,
                    fidelity=num_instance_per_class,
                    seed=self.seed,
                )

                compute_time = time.time() - start

                # tracking
                self.total_time += compute_time
                self.best_score = max(
                    self.best_score,
                    score,
                )
                self.evaluation_count += 1

                self.results.add(
                    fidelity=fidelity,
                    config=config,
                    score=score,
                    train_samples=fidelity,
                    num_instance_per_class=num_instance_per_class,
                    compute_time=compute_time,
                    cumulative_time=self.total_time,
                    best_score_so_far=self.best_score,
                    evaluation_count=self.evaluation_count,
                )

                print(f"Fidelity {fidelity}: " f"{score:.4f}")

                trial.report(
                    score,
                    step=fidelity,
                )

                if trial.should_prune():

                    print("Trial pruned at fidelity:", fidelity)

                    raise optuna.TrialPruned()

                final_score = score

            return final_score

        try:
            study = bohb.optimize(objective)
        finally:
            print("\nCleaning embedding cache...")
            self.evaluator.cache.clear()

        best_score = study.best_value
        best_config = study.best_params

        self.results.save_best(
            best_score,
            best_config,
            cumulative_time=self.total_time,
            evaluation_count=self.evaluation_count,
        )

        return best_score, best_config

    def fit(self):

        if self.search_strategy == "successive_halving":

            best_score, best_config = self.successive_halving()

        elif self.search_strategy == "bohb":

            best_score, best_config = self.bohb()

        else:

            raise ValueError(f"Unknown search strategy: {self.search_strategy}")

        print("\n" + "=" * 80)
        print("AutoML Search Completed")
        print("=" * 80)

        print(f"Best Score: {best_score:.4f}")
        print("Best Config:")
        print(best_config)

        return best_score, best_config
