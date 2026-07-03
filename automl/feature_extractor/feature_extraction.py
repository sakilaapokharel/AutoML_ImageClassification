"""Feature Extraction box: turns a dataset variant into a flat dict of
'dataset meta-features' -- these become the 'dataset features' columns in the
final results dataframe (dataset features | model | augmentation | loss)."""
import numpy as np
from collections import Counter
from PIL import Image
from automl.utils import get_targets


def extract_dataset_meta_features(dataset, variant_name="original"):
    targets = get_targets(dataset)
    counts = Counter(targets)
    n = len(targets)

    class_counts = np.array(list(counts.values()), dtype=float)
    probs = class_counts / class_counts.sum()

    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    imbalance_ratio = float(class_counts.max() / class_counts.min())

    # -----------------------------
    # Image statistics
    # -----------------------------
    widths, heights = [], []
    pixel_means, pixel_stds = [], []

    for i in range(min(len(dataset), 1000)):  # cap for speed
        img = dataset[i][0]  # assumes (image, label)

        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))

        arr = np.array(img)

        # shape handling (H, W) or (H, W, C)
        if arr.ndim == 2:
            h, w = arr.shape
            arr = np.stack([arr] * 3, axis=-1)
        else:
            h, w, _ = arr.shape

        widths.append(w)
        heights.append(h)

        pixel_means.append(arr.mean())
        pixel_stds.append(arr.std())

    img_mean_width = float(np.mean(widths))
    img_mean_height = float(np.mean(heights))
    img_mean_pixel_value = float(np.mean(pixel_means))
    img_std_pixel_value = float(np.mean(pixel_stds))

    return {
        "variant": variant_name,
        "n_samples": n,
        "n_classes": len(counts),

        "min_class_count": int(class_counts.min()),
        "max_class_count": int(class_counts.max()),
        "mean_class_count": float(class_counts.mean()),
        "class_count_std": float(class_counts.std()),

        "class_balance_entropy": entropy,
        "imbalance_ratio": imbalance_ratio,

        # image features
        "img_mean_width": img_mean_width,
        "img_mean_height": img_mean_height,
        "img_mean_pixel_value": img_mean_pixel_value,
        "img_std_pixel_value": img_std_pixel_value,
    }
