import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research.automl_cnn_tabpfn.feature_extractor import build_backbone, build_transforms, get_device
from research.automl_cnn_tabpfn.distill import MLPHead


def load_embed_model(tag, save_dir="saved_models_embed"):
    save_dir = Path(save_dir)
    meta_path = save_dir / f"{tag}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No saved embedding model found for tag '{tag}' in {save_dir}"
        )

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

    context_data = np.load(save_dir / f"{tag}_tabpfn_context.npz")
    X_context, y_context = context_data["X_context"], context_data["y_context"]

    from tabpfn_extensions import TabPFNClassifier
    from tabpfn_extensions.embedding import TabPFNEmbedding

    clf = TabPFNClassifier(n_estimators=1, device="cpu")
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)
    embedder.fit(X_context, y_context)

    mlp = MLPHead(in_dim=meta["in_dim"], num_classes=meta["num_classes"])
    mlp.load_state_dict(torch.load(save_dir / f"{tag}_mlp.pt", map_location="cpu"))
    mlp.eval()

    _, eval_transform, _ = build_transforms(
        resize_size=meta["resize_size"], augment=False
    )

    return {
        "cnn": cnn,
        "pca": pca,
        "embedder": embedder,
        "X_context": X_context,
        "y_context": y_context,
        "mlp": mlp,
        "transform": eval_transform,
        "device": device,
        "meta": meta,
    }


@torch.no_grad()
def predict_image(image_path, model_bundle):
    cnn = model_bundle["cnn"]
    pca = model_bundle["pca"]
    embedder = model_bundle["embedder"]
    X_context = model_bundle["X_context"]
    y_context = model_bundle["y_context"]
    mlp = model_bundle["mlp"]
    transform = model_bundle["transform"]
    device = model_bundle["device"]

    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    feat = cnn(img_t).cpu().numpy()
    if pca is not None:
        feat = pca.transform(feat)

    embed = embedder.get_embeddings(X_context, y_context, feat, data_source="test")
    embed = embed.mean(axis=0)

    logits = mlp(torch.tensor(embed, dtype=torch.float32))
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_class = int(probs.argmax().item())
    confidence = float(probs[pred_class].item())

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "all_probs": probs.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--save_dir", default="saved_models_embed")
    args = parser.parse_args()

    model_bundle = load_embed_model(args.tag, save_dir=args.save_dir)
    result = predict_image(args.image, model_bundle)
    print(
        f"{args.image}: class={result['predicted_class']} (confidence={result['confidence']:.4f})"
    )
    print(f"All class probabilities: {result['all_probs']}")


if __name__ == "__main__":
    main()
