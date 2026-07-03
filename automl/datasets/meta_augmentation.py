"""
Meta-augmentation strategies.

Supports:
- single strategy runs
- random multi-strategy compositions per run
- special multi-output strategy: train_imbalanced_and_balanced

Each run yields:
    (run_id, tag, dataset_variant)
"""

import random
from collections import defaultdict
from torch.utils.data import Subset

from automl.utils import get_targets


# =========================================================
# Strategy registry
# =========================================================

STRATEGIES = [
    "original",
    "drop_pct_classes",
    "drop_imbalance",
    "balanced",
    "scarce_data",
    "train_imbalanced_and_balanced",
]


# =========================================================
# Utilities
# =========================================================

def _indices_by_class(dataset):
    targets = get_targets(dataset)
    by_class = defaultdict(list)
    for idx, t in enumerate(targets):
        by_class[t].append(idx)
    return by_class


# =========================================================
# Base strategies
# =========================================================

def original(dataset, seed=0):
    return Subset(dataset, list(range(len(dataset)))), "original"


def drop_random_pct_classes(
    dataset,
    seed=0,
    allowed_pcts=(10, 20, 30, 40, 50, 60, 70, 80, 90),
    max_tries=20
):
    rng = random.Random(seed)

    by_class = _indices_by_class(dataset)
    classes = list(by_class.keys())

    if len(classes) <= 2:
        return dataset, "no_drop_too_few_classes"

    for _ in range(max_tries):
        pct = rng.choice(allowed_pcts)

        num_drop = int(round((pct / 100) * len(classes)))
        num_keep = len(classes) - num_drop

        if num_keep < 2:
            continue

        rng.shuffle(classes)
        keep_classes = set(classes[:num_keep])

        if len(keep_classes) < 2:
            continue

        indices = [i for c in keep_classes for i in by_class[c]]
        return Subset(dataset, indices), f"drop_{pct}pct_classes"

    keep_classes = set(classes[:2])
    indices = [i for c in keep_classes for i in by_class[c]]
    return Subset(dataset, indices), "fallback_keep_2_classes"


def drop_imbalance(dataset, seed=0, min_frac=0.1, max_frac=1.0):
    rng = random.Random(seed)

    by_class = _indices_by_class(dataset)
    indices = []

    for idxs in by_class.values():
        frac = rng.uniform(min_frac, max_frac)
        k = max(1, int(round(len(idxs) * frac)))
        indices.extend(rng.sample(idxs, k))

    return Subset(dataset, indices), "drop_imbalance"


def balanced_subset(dataset, seed=0):
    rng = random.Random(seed)

    by_class = _indices_by_class(dataset)
    min_count = min(len(v) for v in by_class.values())

    indices = []
    for idxs in by_class.values():
        indices.extend(rng.sample(idxs, min_count))

    return Subset(dataset, indices), "balanced"


def scarce_data(dataset, seed=0, fraction=0.1):
    rng = random.Random(seed)

    n = len(dataset)
    k = max(1, int(round(n * fraction)))
    indices = rng.sample(range(n), k)

    return Subset(dataset, indices), "scarce_data"


def train_imbalanced_and_balanced(dataset, seed=0):
    imb, _ = drop_imbalance(dataset, seed=seed)
    bal, _ = balanced_subset(dataset, seed=seed)

    return [
        (imb, "imbalanced_variant"),
        (bal, "balanced_variant"),
    ]


# =========================================================
# Strategy mapping
# =========================================================

STRATEGY_FNS = {
    "original": original,
    "drop_pct_classes": drop_random_pct_classes,
    "drop_imbalance": drop_imbalance,
    "balanced": balanced_subset,
    "scarce_data": scarce_data,
}


# =========================================================
# Composition logic
# =========================================================

def apply_strategies(dataset, strategies, seed):
    current = dataset
    tags = []

    for i, s in enumerate(strategies):
        fn = STRATEGY_FNS[s]
        current, tag = fn(current, seed=seed + i)
        tags.append(tag)

    return current, "+".join(tags)


# =========================================================
# MAIN GENERATOR (STRICT ORDERING)
# =========================================================

def generate_meta_augmentations(
    dataset,
    n_runs=5,
    seed_base=0,
    max_combo_size=3,
    combo_prob=0.5,
    force_original_last=True,
):
    """
    STRICT GUARANTEES:
    - original ONLY appears at last index
    - balanced NEVER appears before last index
    - no leakage from multi-output strategy
    """

    rng = random.Random(seed_base)
    run_id = 0

    # -----------------------------------------------------
    # SAFE POOL (NO original, NO balanced, NO composite)
    # -----------------------------------------------------
    safe_pool = [
        s for s in STRATEGY_FNS.keys()
        if s not in {"original"}
    ]

    total_random_runs = n_runs - 1 if force_original_last else n_runs

    # =====================================================
    # 1. RANDOM PHASE (STRICT)
    # =====================================================
    for i in range(total_random_runs):
        seed = seed_base + i

        if rng.random() < combo_prob:
            k = rng.randint(2, min(max_combo_size, len(safe_pool)))
            strategies = rng.sample(safe_pool, k)
        else:
            strategies = [rng.choice(safe_pool)]

        variant, tag = apply_strategies(dataset, strategies, seed)

        # HARD SAFETY: prevent accidental leakage
        if "balanced" in tag:
            continue
        if tag == "original":
            continue

        yield run_id, tag, variant
        run_id += 1

    # =====================================================
    # 2. FORCE ORIGINAL LAST
    # =====================================================
    if force_original_last:
        yield run_id, "original_forced_last", Subset(
            dataset, list(range(len(dataset)))
        )