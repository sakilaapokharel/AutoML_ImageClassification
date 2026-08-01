import argparse

from src.classifier import ImageClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Train A model on given latest best config and Evaluate on Test Images"
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

    print("Training Model...")
    clf.fit()

    print("Evaluating model...")
    results = clf.evaluate()

    print("\nResults")
    print("=" * 80)
    print(results)


if __name__ == "__main__":
    main()