# tabpfn_portfolio.py

import os
import pandas as pd

from dotenv import load_dotenv
from tabpfn import TabPFNRegressor

from automl_tabpfn.get_features import get_features_and_target
from automl_tabpfn.preprocess import preprocess
from automl_metadata.feature_extractor.feature_extraction import (
    extract_dataset_meta_features,
)
from automl_metadata.datasets.dataset_loader import load_dataset
from automl_tabpfn.get_model_search_space import generate_test_configurations

CONFIG_COLUMNS = [
    "model",
    "sampler",
    "resize",
    "augmentation",
    "loss",
]


def run_tabpfn_portfolio(dataset_name, data_root, target, k=1):

    load_dotenv()

    os.environ["TABPFN_TOKEN"] = os.getenv("TABPFN_API_KEY")

    if target == "accuracy":
        target_col = "test_accuracy"
    else:
        target_col = "compute_time_sec"

    # -----------------------
    # Train TabPFN
    # -----------------------

    df = pd.read_csv("metadatas/combined_metadata_csv.csv")

    X_train, y_train, feat_cols = get_features_and_target(df, target)

    X_train_processed, preprocessor = preprocess(X_train)

    model = TabPFNRegressor(random_state=42)

    model.fit(X_train_processed, y_train)

    # -----------------------
    # Generate candidates
    # -----------------------

    dataset = load_dataset(dataset_name, data_root=data_root)

    dataset_features = extract_dataset_meta_features(dataset.train)

    X_test = generate_test_configurations(dataset_features)

    X_test = X_test[feat_cols]

    X_test_processed = preprocessor.transform(X_test)

    predictions = model.predict(X_test_processed)

    X_test[target_col] = predictions

    # -----------------------
    # Select top-k
    # -----------------------

    if target == "accuracy":

        X_test = X_test.sort_values(target_col, ascending=False)

    else:

        X_test = X_test.sort_values(target_col, ascending=True)

    portfolio = []

    for _, row in X_test.head(k).iterrows():

        config = {c: row[c] for c in CONFIG_COLUMNS}

        portfolio.append(config)

    return portfolio
