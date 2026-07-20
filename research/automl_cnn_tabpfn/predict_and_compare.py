import argparse
import json
from pathlib import Path

from research.automl_cnn_tabpfn.configs import DATASET_DIRS
from research.automl_cnn_tabpfn.data_utils import load_csv
from research.automl_cnn_tabpfn.predict import load_distilled_model, predict_image


def resolve_image_path(images_dir, fname):
    p = Path(images_dir) / fname
    if p.exists():
        return p
    for ext in (".jpg", ".jpeg", ".png"):
        cand = Path(images_dir) / f"{fname}{ext}"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find image for '{fname}' in {images_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag", required=True, help="model tag, e.g. emotions_resnet18_n11_aug0"
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASET_DIRS.keys()))
    parser.add_argument("--save_dir", default="saved_models")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional cap on number of test images to evaluate (useful for a quick check)",
    )
    args = parser.parse_args()

    dataset_dir = DATASET_DIRS[args.dataset]
    test_df, file_col, label_col = load_csv(dataset_dir / "test.csv")

    if args.limit:
        test_df = test_df.head(args.limit)

    print(f"Loading model '{args.tag}'...")
    model_bundle = load_distilled_model(args.tag, save_dir=args.save_dir)

    images_dir = dataset_dir / "images_test"

    image_file_names = []
    true_labels = []
    predicted_labels = []
    corrects = []
    confidences = []
    correct = 0
    total = 0

    print(f"Running predictions on {len(test_df)} test images...")
    for i, row in test_df.iterrows():
        fname = str(row[file_col])
        true_label = int(row[label_col])

        try:
            img_path = resolve_image_path(images_dir, fname)
            pred = predict_image(img_path, model_bundle)
        except FileNotFoundError as e:
            print(f"⚠️ Skipping {fname}: {e}")
            continue

        is_correct = pred["predicted_class"] == true_label
        correct += int(is_correct)
        total += 1

        image_file_names.append(fname)
        true_labels.append(true_label)
        predicted_labels.append(pred["predicted_class"])
        corrects.append(is_correct)
        confidences.append(pred["confidence"])

        if total % 500 == 0:
            print(f"  ...{total} images processed")

    accuracy = correct / total if total > 0 else 0.0

    output = {
        "tag": args.tag,
        "dataset": args.dataset,
        "num_images": total,
        "num_correct": correct,
        "accuracy": accuracy,
        "columns": {
            "image_file_name": image_file_names,
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "correct": corrects,
            "confidence": confidences,
        },
    }

    out_dir = Path("prediction_comparisons")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.tag}_{args.dataset}_compare.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Saved comparison results to {out_path}")


if __name__ == "__main__":
    main()
