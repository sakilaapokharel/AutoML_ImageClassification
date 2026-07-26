import json
from pathlib import Path

import pandas as pd


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


def find_successive_halving_records(dataset, graphs_root="graphs"):
    """Scans every subfolder under graphs/<dataset>/, and returns the combined
    records ONLY from folders whose records identify as 'successive_halving'."""
    dataset_root = Path(graphs_root) / dataset
    sh_records = []

    for subfolder in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        records = load_run_records(subfolder)
        if not records:
            continue

        strategies_seen = [r.get("search_strategy") for r in records if r.get("search_strategy")]
        if not strategies_seen:
            continue

        majority_strategy = max(set(strategies_seen), key=strategies_seen.count)
        if majority_strategy == "successive_halving":
            sh_records.extend(records)

    return sh_records


def record_to_row(record):
    config = record.get("config", {})
    embedding_dim = config.get("embedding_dim")
    pca_value = embedding_dim if embedding_dim is not None else "No PCA"

    return {
        "encoder": config.get("encoder"),
        "PCA": pca_value,
        "augmentation": config.get("preprocess_policy"),
        "accuracy": round(record.get("accuracy"),4),
    }


def records_to_df(records):
    rows = [record_to_row(r) for r in records if r.get("accuracy") is not None]
    return pd.DataFrame(rows)


def table_top_unique_encoders(records, top_n=3):
    df = records_to_df(records)
    if df.empty:
        return df
    best_per_encoder = df.loc[df.groupby("encoder")["accuracy"].idxmax()]
    return best_per_encoder.sort_values("accuracy", ascending=False).head(top_n).reset_index(drop=True)


def table_top_overall(records, top_n=3):
    df = records_to_df(records)
    if df.empty:
        return df
    return df.sort_values("accuracy", ascending=False).head(top_n).reset_index(drop=True)


def table_top_configs_for_best_encoder(records, top_n=3):
    df = records_to_df(records)
    if df.empty:
        return df, None
    best_encoder = df.loc[df["accuracy"].idxmax(), "encoder"]
    encoder_df = df[df["encoder"] == best_encoder]
    top_configs = encoder_df.sort_values("accuracy", ascending=False).head(top_n).reset_index(drop=True)
    return top_configs, best_encoder


def save_table_csv(df, out_path):
    out_path = Path(out_path)

    if df.empty:
        with open(out_path, "w") as f:
            f.write("encoder,pca,augmentation,accuracy\n")
        print(f"Saved (empty) to {out_path}")
        return

    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


def generate_interpretability_tables(dataset, graphs_root="graphs", top_n=3):
    dataset_root = Path(graphs_root) / dataset
    sh_records = find_successive_halving_records(dataset, graphs_root=graphs_root)

    if not sh_records:
        print(f"No successive_halving records found for dataset '{dataset}'")
        return

    df1 = table_top_unique_encoders(sh_records, top_n=top_n)
    save_table_csv(df1, dataset_root / f"{dataset}_top3_unique_encoders.csv")

    df2 = table_top_overall(sh_records, top_n=top_n)
    save_table_csv(df2, dataset_root / f"{dataset}_top3_overall.csv")

    df3, best_encoder = table_top_configs_for_best_encoder(sh_records, top_n=top_n)
    save_table_csv(df3, dataset_root / f"{dataset}_top3_best_encoder_configs.csv")


if __name__ == "__main__":
    generate_interpretability_tables("skin_cancer", top_n=5)