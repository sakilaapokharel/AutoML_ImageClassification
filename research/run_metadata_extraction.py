"""CLI entry point.

Example:
    python main.py --data_root data --datasets emotions --n_meta_augmentations 5 \\
        --max_epochs 9 --device cuda --output_csv automl_results.csv
"""

import argparse
from research.automl_metadata.pipeline import run_pipeline
from research.automl_metadata.utils import merge_all_csv
import random


def main():
    p = argparse.ArgumentParser(
        description="End-to-end AutoML pipeline: dataset -> meta-augmentation -> "
        "feature extraction -> Hyperband training over "
        "{model x augmentation x loss}."
    )
    p.add_argument(
        "--data_root",
        default="data",
        help="Folder containing one subfolder per dataset, e.g. data/emotions",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of dataset folder names to run. Default: all found under data_root.",
    )
    p.add_argument(
        "--n_meta_augmentations",
        type=int,
        default=5,
        help="Number of random meta-augmentation iterations per dataset "
        "(the diagram's 'Run N different iterations').",
    )
    p.add_argument(
        "--max_epochs",
        type=int,
        default=11,
        help="Hyperband max resource (epochs given to top-surviving configs).",
    )
    p.add_argument("--eta", type=int, default=3, help="Hyperband downsampling rate.")
    p.add_argument("--device", default="cpu", help="'cpu' or 'cuda'.")
    p.add_argument(
        "--filename_col",
        default=None,
        help="Override auto-detected filename column in train.csv/test.csv.",
    )
    p.add_argument(
        "--label_col",
        default=None,
        help="Override auto-detected label column in train.csv/test.csv.",
    )
    p.add_argument("--output_dir", default=f"metadatas/")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed = random.randint(1, 111)
    run_pipeline(
        data_root=args.data_root,
        dataset_names=args.datasets,
        n_meta_augmentations=args.n_meta_augmentations,
        max_epochs=args.max_epochs,
        eta=args.eta,
        device=args.device,
        filename_col=args.filename_col,
        label_col=args.label_col,
        output_dir=args.output_dir,
        seed=seed,
    )

    # call after pipeline
    merge_all_csv(args.output_dir)

    print("End of Metadata Collection")


if __name__ == "__main__":
    main()
