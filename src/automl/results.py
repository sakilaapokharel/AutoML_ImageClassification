import json
from pathlib import Path
from datetime import datetime


class Results:

    def __init__(
        self,
        dataset_name,
        tabpfn_mode,
        search_strategy,
        seed,
        root="results",
    ):

        self.dataset_name = dataset_name
        self.tabpfn_mode = tabpfn_mode
        self.seed = seed
        self.search_strategy = search_strategy

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_name = f"{dataset_name}_{timestamp}_{seed}"

        self.output_dir = Path(root) / self.run_name

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("Saving results to:", self.output_dir.absolute())

        self.results = []

    def add(
        self,
        fidelity,
        config,
        score,
        train_samples,
        num_instance_per_class,
        compute_time,
        best_score_so_far,
        cumulative_time,
        evaluation_count,
    ):

        result = {
            "fidelity": fidelity,
            "config": config,
            "accuracy": float(score),
            "train_samples": train_samples,
            "num_instance_per_class": num_instance_per_class,
            "search_strategy": self.search_strategy,
            "best_score_so_far": best_score_so_far,
            "cumulative_time": cumulative_time,
            "evaluation_count": evaluation_count,
            "compute_time": round(
                compute_time,
                3,
            ),
        }

        self.results.append(result)

        # save immediately
        self.save_stage(fidelity)

    def save_stage(
        self,
        fidelity,
    ):

        stage_results = [r for r in self.results if r["fidelity"] == fidelity]

        file = self.output_dir / f"{self.run_name}_f{fidelity}.json"

        data = {
            "dataset": self.dataset_name,
            "tabpfn_mode": self.tabpfn_mode,
            "seed": self.seed,
            "fidelity": fidelity,
            "results": stage_results,
            "search_strategy": self.search_strategy,
        }

        with open(file, "w") as f:
            json.dump(
                data,
                f,
                indent=4,
            )

        print(f"Saved {len(stage_results)} results -> {file}")

    def save_best(
        self,
        score,
        config,
        cumulative_time,
        evaluation_count,
    ):

        file = self.output_dir / "best.json"

        with open(file, "w") as f:

            json.dump(
                {
                    "dataset": self.dataset_name,
                    "tabpfn_mode": self.tabpfn_mode,
                    "seed": self.seed,
                    "best_accuracy": float(score),
                    "best_config": config,
                    "search_strategy": self.search_strategy,
                    "cumulative_time": cumulative_time,
                    "evaluation_count": evaluation_count,
                },
                f,
                indent=4,
            )

        print(f"Saved best config -> {file}")

    def best(self):

        if not self.results:
            return None

        best = max(
            self.results,
            key=lambda x: x["accuracy"],
        )

        return (
            best["accuracy"],
            best["config"],
        )

    def all(self):

        return self.results
