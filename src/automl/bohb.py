import optuna
import time


class BOHBSearch:

    def __init__(
        self,
        search_space,
        fidelities,
        reduction_factor,
        n_trials,
        seed,
    ):
        self.search_space = search_space
        self.fidelities = fidelities
        self.reduction_factor = reduction_factor
        self.n_trials = n_trials
        self.seed = seed

    def create_study(self):

        pruner = optuna.pruners.HyperbandPruner(
            min_resource=self.fidelities[0],
            max_resource=self.fidelities[-1],
            reduction_factor=self.reduction_factor,
        )

        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            multivariate=True,
        )

        return optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

    def suggest_config(
        self,
        trial,
    ):

        return {
            "encoder": trial.suggest_categorical(
                "encoder",
                self.search_space["encoder"],
            ),
            "embedding_dim": trial.suggest_categorical(
                "embedding_dim",
                self.search_space["embedding_dim"],
            ),
            "resize": 224,
            "preprocess_policy": trial.suggest_categorical(
                "preprocess_policy",
                self.search_space["preprocess_policy"],
            ),
        }

    def optimize(
        self,
        objective,
    ):

        study = self.create_study()

        print("\n" + "=" * 80)
        print("Starting BOHB Search")
        print("=" * 80)

        print(f"Trials                  : {self.n_trials}")
        print(f"Min fidelity            : {self.fidelities[0]}")
        print(f"Max fidelity            : {self.fidelities[-1]}")
        print(f"Reduction factor        : {self.reduction_factor}")

        print("=" * 80)

        def wrapped_objective(trial):

            print("\n" + "-" * 70)
            print(f"Running Trial {trial.number + 1}/{self.n_trials}")
            print("-" * 70)

            config = self.suggest_config(trial)

            print("Configuration:")
            print(config)

            start_trial = time.time()

            try:

                score = objective(
                    trial,
                    config,
                )

            except optuna.TrialPruned:

                elapsed = time.time() - start_trial

                print("\nTrial pruned")

                print(f"Trial time: {elapsed:.2f}s")

                raise

            elapsed = time.time() - start_trial

            print("\nTrial finished")

            print(f"Score : {score:.4f}")

            print(f"Time  : {elapsed:.2f}s")

            return score

        study.optimize(
            wrapped_objective,
            n_trials=self.n_trials,
        )

        print("\n" + "=" * 80)
        print("BOHB Finished")
        print("=" * 80)

        print(f"Best Score : {study.best_value:.4f}")

        print("Best Config:")
        print(study.best_params)

        return study
