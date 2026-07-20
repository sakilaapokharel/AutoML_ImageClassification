import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from research.automl_cnn_tabpfn.configs import DATASET_DIRS
from research.automl_cnn_tabpfn.data_utils import ImageCSVDataset, load_csv
from research.automl_cnn_tabpfn.feature_extractor import extract_features
from research.automl_cnn_tabpfn.predict_embed import load_embed_model
from research.automl_cnn_tabpfn.train_embed import batched_get_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dataset", required=True, choices=list(DATASET_DIRS.keys()))
    parser.add_argument("--save_dir", default="saved_models_embed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_batch_size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset_dir = DATASET_DIRS[args.dataset]
    test_df, file_col, label_col = load_csv(dataset_dir / "test.csv")
    if args.limit:
        test_df = test_df.head(args.limit)

    print(f"Loading model '{args.tag}'...")
    model_bundle = load_embed_model(args.tag, save_dir=args.save_dir)
    cnn = model_bundle["cnn"]
    pca = model_bundle["pca"]
    embedder = model_bundle["embedder"]
    X_context = model_bundle["X_context"]
    y_context = model_bundle["y_context"]
    mlp = model_bundle["mlp"]
    transform = model_bundle["transform"]
    device = model_bundle["device"]

    print("Extracting CNN features for test images...")
    test_ds = ImageCSVDataset(
        test_df, file_col, label_col, dataset_dir / "images_test", transform
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    X_test, y_test = extract_features(cnn, test_loader, device)

    if pca is not None:
        X_test = pca.transform(X_test)

    print("Extracting TabPFN embeddings for test set...")
    test_embed = batched_get_embeddings(
        embedder, X_context, y_context, X_test, batch_size=args.embed_batch_size
    )
    test_embed = test_embed.mean(axis=0)

    mlp.eval()
    with torch.no_grad():
        logits = mlp(torch.tensor(test_embed, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).numpy()
        confidences = probs.max(dim=1).values.numpy()

    correct_mask = preds == y_test
    accuracy = float(correct_mask.mean())

    output = {
        "tag": args.tag,
        "dataset": args.dataset,
        "num_images": len(y_test),
        "num_correct": int(correct_mask.sum()),
        "accuracy": accuracy,
        "columns": {
            "image_file_name": test_df[file_col].astype(str).tolist(),
            "true_label": y_test.tolist(),
            "predicted_label": preds.tolist(),
            "correct": correct_mask.tolist(),
            "confidence": confidences.tolist(),
        },
    }

    out_dir = Path("prediction_comparisons_embed")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.tag}_{args.dataset}_compare.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nAccuracy: {accuracy:.4f} ({int(correct_mask.sum())}/{len(y_test)})")
    print(f"Saved comparison results to {out_path}")


if __name__ == "__main__":
    main()
