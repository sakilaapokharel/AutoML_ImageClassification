import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_run_records(folder_path):
    """Loads every .json file in a folder and returns a combined flat list of records."""
    folder_path = Path(folder_path)
    all_records = []

    for json_file in sorted(folder_path.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        if isinstance(data, list):
            all_records.extend(data)
        elif isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list):
                all_records.extend(data["results"])
            elif "records" in data and isinstance(data["records"], list):
                all_records.extend(data["records"])
            else:
                all_records.append(data)

    return all_records


def discover_strategy_folders(dataset_root):
    """Scans every subfolder of dataset_root, reads the JSON records inside,
    and groups folders by the 'search_strategy' value found in their records.
    Returns {strategy_name: [combined records across all folders with that strategy]}."""
    dataset_root = Path(dataset_root)
    strategy_records = {}

    for subfolder in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        records = load_run_records(subfolder)
        if not records:
            continue

        # majority-vote the strategy label found inside this folder's records
        strategies_seen = [r.get("search_strategy") for r in records if r.get("search_strategy")]
        if not strategies_seen:
            print(f"Warning: no 'search_strategy' field found in {subfolder}, skipping")
            continue

        strategy = max(set(strategies_seen), key=strategies_seen.count)
        strategy_records.setdefault(strategy, []).extend(records)

    return strategy_records


def records_to_anytime_curve(records, x_key="cumulative_time", y_key="best_score_so_far"):
    """Converts a list of records into a monotonic 'best accuracy achieved by time X' curve."""
    points = [(r[x_key], r[y_key]) for r in records if x_key in r and y_key in r]
    points.sort(key=lambda p: p[0])

    xs, ys = [], []
    running_best = float("-inf")
    for x, y in points:
        running_best = max(running_best, y)
        xs.append(x)
        ys.append(running_best)

    return xs, ys


def plot_accuracy_vs_time(
    dataset,
    graphs_root="graphs",
    x_key="cumulative_time",
    y_key="best_score_so_far",
    xlabel="cumulative_time",
    ylabel="best_score_so_far",
    save_path=None,
    figsize=(8, 5),
):
    """Give just the dataset name -- it auto-discovers strategy subfolders
    (e.g. graphs/<dataset>/*) and groups them by 'search_strategy' found
    inside the JSON records, regardless of how the folders are named."""
    dataset_root = Path(graphs_root) / dataset
    strategy_records = discover_strategy_folders(dataset_root)

    if not strategy_records:
        print(f"No strategy records found under {dataset_root}")
        return

    plt.figure(figsize=figsize)

    for strategy, records in strategy_records.items():
        xs, ys = records_to_anytime_curve(records, x_key=x_key, y_key=y_key)
        plt.plot(xs, ys, label=strategy)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{dataset.capitalize()}: {' vs '.join(strategy_records.keys())}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = dataset_root / f"{dataset}_bohb_vs_sh.png"

    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.show()


# ============================================================
# RUN — just give the dataset name
# ============================================================
if __name__ == "__main__":
    plot_accuracy_vs_time("skin_cancer")