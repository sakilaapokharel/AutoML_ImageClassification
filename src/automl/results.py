import json
from pathlib import Path
from datetime import datetime


class Results:

    def __init__(
        self,
        dataset_name,
        tabpfn_mode,
        seed,
        save_dir="results",
    ):

        self.metadata = {
            "dataset": dataset_name,
            "tabpfn_mode": tabpfn_mode,
            "seed": seed,
        }

        self.results = []

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.path = self.save_dir / f"{dataset_name}_{timestamp}.json"

    def add(
        self,
        config,
        score,
        train_samples,
        test_samples,
        compute_time,
    ):

        self.results.append(
            {
                "config": config,
                "train_samples": train_samples,
                "test_samples": test_samples,
                "test_accuracy": float(score),
                "compute_time_seconds": float(compute_time),
            }
        )

        self.save()

    def save(self):

        output = {
            **self.metadata,
            "experiments": self.results,
        }

        with open(self.path, "w") as f:
            json.dump(
                output,
                f,
                indent=4,
            )

        print(f"Results updated: {self.path}")

    def best(self):

        return max(self.results, key=lambda x: x["test_accuracy"])

    def all(self):

        return self.results
