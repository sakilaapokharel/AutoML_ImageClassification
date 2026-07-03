"""Meta-augmentation strategies -- the branches coming out of 'Dataset loaders'
in the diagram:

    - original dataset also
    - Randomly drop 50% class
    - Drop imbalance dataset
    - Train imbalance dataset and one balanced dataset
    - Scarce data

`generate_meta_augmentations` implements the 'Run N different iterations' box:
each iteration randomly picks one of the five strategies above and applies it
with a fresh seed, so you get N randomly-varied dataset variants to push
through feature extraction + Hyperband.
"""
import random
from collections import defaultdict
from torch.utils.data import Subset

from automl.utils import get_targets

STRATEGIES = [
    "original",
    "drop_pct_classes",
    "drop_imbalance",
    "train_imbalanced_and_balanced",
    "scarce_data",
]


def _indices_by_class(dataset):
    targets = get_targets(dataset)
    by_class = defaultdict(list)
    for idx, t in enumerate(targets):
        by_class[t].append(idx)
    return by_class


def original(dataset, seed=0):
    return Subset(dataset, list(range(len(dataset)))), "original"


def drop_random_pct_classes(dataset, seed=0, allowed_pcts=(10, 20, 30, 40, 50, 60, 70, 80, 90), max_tries=20):
    """
    Randomly drop classes by a percentage, ensuring at least 2 classes remain.
    """

    rng = random.Random(seed)

    by_class = _indices_by_class(dataset)
    classes = list(by_class.keys())
    num_classes = len(classes)

    if num_classes <= 2:
        return dataset, "no_drop_too_few_classes"

    for _ in range(max_tries):

        pct = rng.choice(allowed_pcts)

        num_drop = int(round((pct / 100) * num_classes))
        num_keep = num_classes - num_drop

        # enforce constraint
        if num_keep < 2:
            continue

        rng.shuffle(classes)
        keep_classes = set(classes[:num_keep])

        # safety check (redundant but robust)
        if len(keep_classes) < 2:
            continue

        indices = [i for c in keep_classes for i in by_class[c]]

        return Subset(dataset, indices), f"drop_{pct}pct_classes"

    # fallback: always valid output
    rng.shuffle(classes)
    keep_classes = set(classes[:2])
    indices = [i for c in keep_classes for i in by_class[c]]

    return Subset(dataset, indices), "fallback_keep_2_classes"


def drop_imbalance(dataset, seed=0, min_frac=0.1, max_frac=1.0):
    """Subsample each class by an independent random fraction, inducing class
    imbalance."""
    rng = random.Random(seed)
    by_class = _indices_by_class(dataset)
    indices = []
    for idxs in by_class.values():
        frac = rng.uniform(min_frac, max_frac)
        k = max(1, int(round(len(idxs) * frac)))
        indices.extend(rng.sample(idxs, k))
    return Subset(dataset, indices), "drop_imbalance"


def balanced_subset(dataset, seed=0):
    """Undersample every class down to the size of the smallest class."""
    rng = random.Random(seed)
    by_class = _indices_by_class(dataset)
    min_count = min(len(v) for v in by_class.values())
    indices = []
    for idxs in by_class.values():
        indices.extend(rng.sample(idxs, min_count))
    return Subset(dataset, indices), "balanced"


def train_imbalanced_and_balanced(dataset, seed=0):
    """'Train imbalance dataset and one balanced dataset' -- returns BOTH variants
    so both get pushed through the rest of the pipeline."""
    imb, _ = drop_imbalance(dataset, seed=seed)
    bal, _ = balanced_subset(dataset, seed=seed)
    return [(imb, "imbalanced_variant"), (bal, "balanced_variant")]


def scarce_data(dataset, seed=0, fraction=0.1):
    """Keep only a small random fraction of the whole dataset."""
    rng = random.Random(seed)
    n = len(dataset)
    k = max(1, int(round(n * fraction)))
    indices = rng.sample(range(n), k)
    return Subset(dataset, indices), "scarce_data"



def generate_meta_augmentations(dataset, n_runs=5, seed_base=0):
    """Run `n_runs` random meta-augmentation iterations (the 'Run N different
    iterations' box). Each iteration randomly selects one of the 5 strategies
    above and applies it with a unique seed.

    Yields (run_id, variant_tag, dataset_variant) tuples. Note
    'train_imbalanced_and_balanced' yields two variants for a single iteration
    (each gets its own run_id).
    """
    rng = random.Random(seed_base)
    run_id = 0
    for i in range(n_runs):
        seed = seed_base + i
        strategy = rng.choice(STRATEGIES)

        if strategy == "train_imbalanced_and_balanced":
            for variant, tag in train_imbalanced_and_balanced(dataset, seed):
                yield run_id, tag, variant
                run_id += 1
            continue

        fn = {
            "original": original,
            "drop_pct_classes": drop_random_pct_classes,
            "drop_imbalance": drop_imbalance,
            "scarce_data": scarce_data,
        }[strategy]
        variant, tag = fn(dataset, seed)
        yield run_id, tag, variant
        run_id += 1
