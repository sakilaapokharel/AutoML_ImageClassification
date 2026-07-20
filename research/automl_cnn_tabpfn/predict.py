import argparse
import json
import pickle
from pathlib import Path

import torch
from PIL import Image

from research.automl_cnn_tabpfn.feature_extractor import build_backbone, build_transforms, get_device
from research.automl_cnn_tabpfn.distill import MLPHead


def load_distilled_model(tag, save_dir="saved_models"):
    save_dir = Path(save_dir)

    meta_path = save_dir / f"{tag}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No saved model found for tag '{tag}' in {save_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    device = get_device()
    cnn, _ = build_backbone(meta["model_name"])
    cnn = cnn.to(device).eval()

    pca = None
    pca_path = save_dir / f"{tag}_pca.pkl"
    if meta["used_pca"] and pca_path.exists():
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)

    mlp = MLPHead(in_dim=meta["in_dim"], num_classes=meta["num_classes"])
    mlp.load_state_dict(torch.load(save_dir / f"{tag}_mlp.pt", map_location="cpu"))
    mlp.eval()

    _, eval_transform, _ = build_transforms(
        resize_size=meta["resize_size"], augment=False
    )

    return {
        "cnn": cnn,
        "pca": pca,
        "mlp": mlp,
        "transform": eval_transform,
        "device": device,
        "meta": meta,
    }


@torch.no_grad()
def predict_image(image_path, model_bundle):
    cnn = model_bundle["cnn"]
    pca = model_bundle["pca"]
    mlp = model_bundle["mlp"]
    transform = model_bundle["transform"]
    device = model_bundle["device"]

    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    feat = cnn(img_t).cpu().numpy()
    if pca is not None:
        feat = pca.transform(feat)

    logits = mlp(torch.tensor(feat, dtype=torch.float32))
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_class = int(probs.argmax().item())
    confidence = float(probs[pred_class].item())

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "all_probs": probs.tolist(),
    }


@torch.no_grad()
def predict_batch(image_paths, model_bundle):
    return [{"image": str(p), **predict_image(p, model_bundle)} for p in image_paths]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag", required=True, help="model tag, e.g. emotions_resnet18_n11_aug0"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="path to a single image, or a directory of images",
    )
    parser.add_argument("--save_dir", default="saved_models")
    args = parser.parse_args()

    model_bundle = load_distilled_model(args.tag, save_dir=args.save_dir)

    image_path = Path(args.image)
    if image_path.is_dir():
        paths = sorted(
            [
                p
                for p in image_path.glob("*")
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
        )
        results = predict_batch(paths, model_bundle)
        for r in results:
            print(
                f"{r['image']}: class={r['predicted_class']} (confidence={r['confidence']:.4f})"
            )
    else:
        result = predict_image(image_path, model_bundle)
        print(
            f"{image_path}: class={result['predicted_class']} (confidence={result['confidence']:.4f})"
        )
        print(f"All class probabilities: {result['all_probs']}")


if __name__ == "__main__":
    main()
