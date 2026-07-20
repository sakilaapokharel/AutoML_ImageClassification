import argparse
from dotenv import load_dotenv
import os

from tabpfn import TabPFNRegressor
import pandas as pd

from research.automl_tabpfn.get_features import get_features_and_target
from research.automl_tabpfn.preprocess import preprocess
from research.automl_metadata.feature_extractor.feature_extraction import (
    extract_dataset_meta_features,
)
from research.automl_metadata.datasets.dataset_loader import load_dataset
from research.automl_tabpfn.get_model_search_space import generate_test_configurations


def main(dataset_name, data_root, target):
    if target == "accuracy":
        target_col = "test_accuracy"
    else:
        target_col = "compute_time_sec"
    load_dotenv()

    print("API key loaded:", os.getenv("TABPFN_API_KEY") is not None)
    os.environ["TABPFN_TOKEN"] = os.getenv("TABPFN_API_KEY")

    # Load training data
    df = pd.read_csv("metadatas/combined_metadata_csv.csv")

    X_train, y_train, feat_cols = get_features_and_target(df, target)

    X_train_processed, preprocessor = preprocess(X_train)

    model = TabPFNRegressor(
        random_state=42,
    )

    model.fit(X_train_processed, y_train)

    # Extract test metadata
    dataset = load_dataset(dataset_name, data_root=data_root)
    dataset_features = extract_dataset_meta_features(dataset.train)
    X_test = generate_test_configurations(dataset_features)

    # Same feature order as training
    X_test = X_test[feat_cols]

    # preprocess
    X_test_processed = preprocessor.transform(X_test)

    # predict
    predictions = model.predict(X_test_processed)

    # Add predictions
    X_test[target_col] = predictions

    # Configuration columns to move to the end
    config_cols = [
        "model",
        "sampler",
        "resize",
        "augmentation",
        "loss",
        "epochs_trained",
    ]

    # Keep all other metadata columns first
    meta_cols = [c for c in X_test.columns if c not in config_cols + [target_col]]

    # Reorder
    X_test = X_test[meta_cols + config_cols + [target_col]]

    # Sort by predicted accuracy
    if target == "accuracy":
        X_test = X_test.sort_values(by=target_col, ascending=False)
    else:
        X_test = X_test.sort_values(by="compute_time_sec", ascending=True)

    # Save
    X_test.to_csv(
        f"outputs/{dataset_name}_{target}_tabpfn_predictions.csv", index=False
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Predict test accuracy using TabPFN")

    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name for feature extraction"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["accuracy", "compute"],
        help="Dataset name for feature extraction",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset name for feature extraction",
    )

    args = parser.parse_args()

    main(args.dataset, args.data_root, args.target)
