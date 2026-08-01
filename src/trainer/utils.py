from pathlib import Path
import json

def get_latest_config(dataset_name):
    results_dir = Path("results")

    prefix = f"{dataset_name}_"

    folders = [
        f for f in results_dir.iterdir() if f.is_dir() and f.name.startswith(prefix)
    ]

    if not folders:
        raise FileNotFoundError(f"No results found for dataset '{dataset_name}'")

    all_results = []

    for folder in folders:
        for json_file in folder.glob("*.json"):

            # skip summary file
            if json_file.name == "best.json":
                continue

            with open(json_file, "r") as f:
                data = json.load(f)

            # append experiment results
            if "results" in data:
                all_results.extend(data["results"])

    return all_results


def get_best_config(configs):
    """
    Select configuration with highest validation accuracy.
    """

    if not configs:
        raise ValueError("No configurations provided")

    return max(configs, key=lambda x: x["accuracy"])
