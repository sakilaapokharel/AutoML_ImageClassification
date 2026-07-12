from automl_tabpfn.config import exclude_columns
import pandas as pd


def get_features_and_target(df: pd.DataFrame, target="accuracy"):
    # Make a copy so we don't modify the global config
    excluded = exclude_columns.copy()

    if target == "accuracy":
        target_column = "test_accuracy"
        excluded.append("compute_time_sec")
    elif target == "compute":
        target_column = "compute_time_sec"
        excluded.append("test_accuracy")
    else:
        raise RuntimeError
    
    feature_columns = [
        col for col in df.columns
        if col not in excluded + [target_column]
    ]

    X_train = df[feature_columns].copy()
    y_train = df[target_column].copy()

    return X_train, y_train, feature_columns