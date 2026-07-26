from search import AutoML
from dotenv import load_dotenv
import argparse
import os


def main():

    load_dotenv()

    parser = argparse.ArgumentParser(description="AutoML Image Classification")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["flowers", "emotions", "fashion", "skin_cancer", "skin_cancer_test"],
        help="Dataset to run AutoML on",
    )

    parser.add_argument(
        "--search_strategy",
        type=str,
        default="successive_halving",
        choices=["successive_halving", "bohb"],
        help=("Number of instances per class. " "Use -1 for all samples."),
    )

    parser.add_argument(
        "--train_fidelity",
        type=int,
        default=-1,
        help=("Number of instances per class. " "Use -1 for all samples."),
    )

    parser.add_argument(
        "--tabpfn_client",
        action="store_true",
        help=("Use TabPFN client API instead of local TabPFN."),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    if args.tabpfn_client:

        print("Using TabPFN client")

        api_key = os.getenv("TABPFN_API_KEY")

        if api_key is None:
            raise RuntimeError("TABPFN_API_KEY not found in environment")

        os.environ["TABPFN_TOKEN"] = api_key

        tabpfn_mode = "client"

    else:

        print("Using local TabPFN")

        tabpfn_mode = "local"

    automl = AutoML(
        dataset=args.dataset,
        seed=args.seed,
        tabpfn_mode=tabpfn_mode,
        train_fidelity=args.train_fidelity,
        search_strategy=args.search_strategy,
    )
    # config = {'encoder': 'efficientnet_b0', 'embedding_dim': None, 'resize': 224, 'augmentation': 'none'}
    # config= {"encoder": "densenet121", "embedding_dim": None, "resize": 224, "augmentation": "none"}
    # best_config = {
    #     "encoder": "tinyvit_5m",
    #     "embedding_dim": 64,
    #     "resize": 224,
    #     "augmentation": "none",
    # }
    # student = automl.distill(
    #     best_config=best_config,
    # )

    best_config = automl.fit()
    # print(best_config)


if __name__ == "__main__":
    main()
