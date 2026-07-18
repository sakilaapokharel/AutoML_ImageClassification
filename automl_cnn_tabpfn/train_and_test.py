import argparse
import json
from pathlib import Path

from configs import DATASET_DIRS, MODEL_CHOICES
from pipeline import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_DIRS.keys()))
    parser.add_argument("--number_instances", type=int, required=True, help="samples per class to train on")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--pca_dim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--augment", action="store_true", help="apply augmentation to training images")
    args = parser.parse_args()

    result = run(
        DATASET_DIRS[args.dataset],
        args.model,
        args.number_instances,
        pca_dim=args.pca_dim,
        seed=args.seed,
        batch_size=args.batch_size,
        resize_size=args.resize_size,
        augment=args.augment,
    )

    print(f"\nDataset: {args.dataset} | Model: {args.model} | Requested N/class: {args.number_instances} "
          f"| Augment: {result['augment']} "
          f"| RandAugment magnitude: {result['randaug_magnitude']} | Resize: {result['resize_size']}")
    print(f"Train size (post-augment): {result['n_train']} | Test size: {result['n_test']} | Embedding dim used: {result['embed_dim']}")
    print(f"Accuracy: {result['accuracy']:.4f}\n")
    print(result["report"])

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.dataset}_{args.model}_n{args.number_instances}_aug_{args.augment}_resize_{args.resize_size}.json"
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "report"}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()