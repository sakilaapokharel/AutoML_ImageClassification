import argparse

from src.search import AutoML
from src.classifier import ImageClassifier


def main():

    parser = argparse.ArgumentParser(
        description="Run AutoML + Distillation Pipeline"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name",
        choices=["emotions", "flowers", "fashion", "skin_cancer"]
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--tabpfn_mode",
        type=str,
        default="local",
        choices=["local", "client"],
        help="TabPFN mode",
    )

    parser.add_argument(
        "--search_strategy",
        type=str,
        default="bohb",
        choices=[
            "successive_halving",
            "bohb",
        ],
        help="AutoML search strategy",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Student inference batch size",
    )

    args = parser.parse_args()


    # --------------------------------------------------
    # 1. AutoML Search
    # --------------------------------------------------

    print("\nRunning AutoML Search")
    print("=" * 80)

    automl = AutoML(
        dataset=args.dataset,
        seed=args.seed,
        tabpfn_mode=args.tabpfn_mode,
        search_strategy=args.search_strategy,
    )

    best_score, best_config = automl.fit()

    print("\nBest AutoML Result")
    print("=" * 80)
    print(f"Score: {best_score:.4f}")
    print(best_config)

    # --------------------------------------------------
    # 2. Distillation Training
    # --------------------------------------------------

    print("\nTraining Image Classifier")
    print("=" * 80)

    clf = ImageClassifier(
        dataset=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        tabpfn_mode=args.tabpfn_mode,
    )

    clf.fit()

    # --------------------------------------------------
    # 3. Evaluation
    # --------------------------------------------------

    print("\nEvaluating")
    print("=" * 80)

    results = clf.evaluate()

    print("\nFinal Results")
    print("=" * 80)
    print(results)


if __name__ == "__main__":
    main()