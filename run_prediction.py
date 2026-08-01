import argparse

from src.classifier import ImageClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate or run inference with a trained ImageClassifier"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name",
        choices=["emotions", "flowers", "fashion", "skin_cancer"]
    )

    args = parser.parse_args()

    clf = ImageClassifier(
        dataset=args.dataset
        )

    print("Loading trained checkpoints...")
    clf.load_checkpoints()

    print("Evaluating model...")
    results = clf.evaluate()

    print("\nResults")
    print("=" * 80)
    print(results)


if __name__ == "__main__":
    main()