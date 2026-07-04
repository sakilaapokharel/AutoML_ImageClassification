"""End-to-end orchestrator:

  select dataset -> Dataset loaders (train.csv/test.csv + images, or
  ImageFolder) -> N random meta-augmentation iterations of the TRAIN set
  -> Feature Extraction -> Hyperband Training (search: model x augmentation
  x loss), always evaluated against the dataset's real, fixed TEST set
  -> save each run as: dataset features | model | augmentation | loss |
     test accuracy | compute time

This is the direct code equivalent of the draw.io diagram.
"""

import random

from automl_metadata.datasets.dataset_loader import (
    list_available_datasets,
    load_dataset,
)
from automl_metadata.datasets.meta_augmentation import generate_meta_augmentations
from automl_metadata.feature_extractor.feature_extraction import (
    extract_dataset_meta_features,
)
from automl_metadata.models.hyperband import hyperband_search
from automl_metadata.models.train_eval import train_one_config

SEARCH_SPACE = {
    "model": [
        "scratch_cnn",
        "resnet18",
        "resnet34",
        "mobilenet_v2",
        "efficientnet_b0",
        "efficientnet_b2",
    ],
    "augmentation": ["rand_aug", "none"],
    "sampler": [
        "random",
        "weighted",
    ],
    "loss": ["cross_entropy", "label_smoothing", "class_balanced"],
}


def sample_config(rng):
    return {
        "model": rng.choice(SEARCH_SPACE["model"]),
        "augmentation": rng.choice(SEARCH_SPACE["augmentation"]),
        "loss": rng.choice(SEARCH_SPACE["loss"]),
        "sampler": rng.choice(SEARCH_SPACE["sampler"]),
    }


def run_pipeline(
    data_root="data",
    dataset_names=None,
    n_meta_augmentations=5,
    max_epochs=9,
    eta=3,
    device="cpu",
    output_dir="metadata/",
    seed=0,
    filename_col=None,
    label_col=None,
    verbose=True,
):
    rng = random.Random(seed)
    available = list_available_datasets(data_root)
    if dataset_names is None:
        dataset_names = available
    else:
        missing = [d for d in dataset_names if d not in available]
        if missing:
            raise ValueError(
                f"Dataset(s) {missing} not found under {data_root}. "
                f"Available: {available}"
            )

    for ds_name in dataset_names:
        bundle = load_dataset(
            ds_name, data_root, filename_col=filename_col, label_col=label_col
        )
        
        num_classes = len(bundle.classes)
        num_channels = bundle.num_channels

        if verbose:
            print(
                f"[{ds_name}] train={len(bundle.train)} test={len(bundle.test)} "
                f"images, {num_classes} classes: {bundle.classes}"
            )

        # Only the TRAIN side gets meta-augmented into variants. The TEST set
        # is fixed and shared across every variant/config/rung.
        for run_id, variant_tag, variant_ds in generate_meta_augmentations(
            bundle.train, n_runs=n_meta_augmentations, seed_base=seed
        ):
            if len(variant_ds) < 4:
                # Too few samples to train/validate meaningfully; skip.
                continue

            meta_features = extract_dataset_meta_features(
                variant_ds, variant_name=variant_tag
            )
            meta_features["dataset"] = ds_name
            meta_features["run_id"] = run_id
            if verbose:
                print(
                    f"  run {run_id} ({variant_tag}): "
                    f"{meta_features['n_samples']} train samples, "
                    f"{meta_features['n_classes']} classes"
                )

            def config_sampler():
                return sample_config(rng)

            def run_config(
                cfg, n_epochs, _variant_ds=variant_ds, _num_classes=num_classes, _num_channels=num_channels
            ):
                return train_one_config(
                    _variant_ds,
                    bundle.test,
                    _num_classes,
                    _num_channels,
                    cfg,
                    n_epochs,
                    device=device,
                    seed=seed,
                )

            hyperband_search(
                config_sampler,
                run_config,
                meta_features=meta_features,
                max_epochs=max_epochs,
                eta=eta,
                output_dir=output_dir,
            )

    return True
