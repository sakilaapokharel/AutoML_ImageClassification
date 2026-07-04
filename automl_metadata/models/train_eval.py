"""
Trains + evaluates a single (model, augmentation, sampler, loss) config.

Key design:
- augmentation → image transform pipeline
- sampler → DataLoader sampling strategy
- loss → optimization objective
- model → architecture selection

Test set is NEVER used for training or model selection.
"""

import time
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm

from automl_metadata.models.models import build_model, trainable_parameters
from automl_metadata.models.losses import get_loss
from automl_metadata.utils import get_targets, TransformedDataset
from automl_metadata.models.sampler import get_sampler 

# =====================================================
# Constants
# =====================================================

IMG_SIZE = 224

RGB_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

GRAY_NORMALIZE = transforms.Normalize(
    mean=[0.5],
    std=[0.5]
)

to_tensor = transforms.ToTensor()

# =====================================================
# Augmentation
# =====================================================

def get_transform(aug_mode, in_channels=3):
    ops = []

    if aug_mode == "none":
        print("No Augmentation Applied")

    elif aug_mode == "rand_aug":
        print("RandAugment Applied")
        mag = random.choice([5, 7, 9, 11, 13, 15])
        ops.append(transforms.RandAugment(num_ops=2, magnitude=mag))

    else:
        raise ValueError(f"Unknown augmentation: {aug_mode}")

    ops.append(transforms.ToTensor())

    if in_channels == 1:
        ops.append(GRAY_NORMALIZE)
    else:
        ops.append(RGB_NORMALIZE)

    return transforms.Compose(ops)


# =====================================================
# Training pipeline
# =====================================================


def train_one_config(
    train_variant,
    test_dataset,
    num_classes,
    num_channels,
    config,
    n_epochs,
    device="cpu",
    val_split=0.15,
    batch_size=32,
    seed=0,
):
    """
    config = {
        "model": ...,
        "augmentation": "none" | "rand_aug",
        "sampler": "none" | "weighted",
        "loss": ...
    }
    """

    # -------------------------------------------------
    # Build transformed dataset
    # -------------------------------------------------
    transform = get_transform(config["augmentation"], num_channels)
    full = TransformedDataset(train_variant, transform)

    # -------------------------------------------------
    # Train / Val split
    # -------------------------------------------------
    targets = get_targets(train_variant)

    g = torch.Generator().manual_seed(0)
    n = len(full)

    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    train_ds, val_ds = torch.utils.data.random_split(
        full, [n_train, n_val], generator=g
    )

    train_targets = [targets[i] for i in train_ds.indices]

    # -------------------------------------------------
    # SAMPLER (NOW CLEANLY SEPARATED)
    # -------------------------------------------------
    sampler, shuffle = get_sampler(
        config.get("sampler", "none"), train_targets, num_classes
    )

    # -------------------------------------------------
    # DataLoaders
    # -------------------------------------------------
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle
    )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    test_transform = get_transform("none", num_channels)
    test_loader = DataLoader(
        TransformedDataset(test_dataset, test_transform),
        batch_size=batch_size,
        shuffle=False,
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = build_model(config["model"], num_classes, in_channels=num_channels).to(
        device
    )

    print("Device:", next(model.parameters()).device)

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------
    class_counts = torch.bincount(
        torch.tensor(train_targets), minlength=num_classes
    ).tolist()

    criterion = get_loss(config["loss"], class_counts=class_counts, device=device)

    optimizer = torch.optim.Adam(trainable_parameters(model), lr=1e-3)

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    start = time.time()
    model.train()

    for epoch in range(n_epochs):
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"[Epoch {epoch+1}] avg_loss={running_loss/len(train_loader):.4f}")

    compute_time = time.time() - start

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------
    def _accuracy(loader):
        correct, total = 0, 0
        model.eval()

        with torch.no_grad():
            for x, y in tqdm(loader, desc="Evaluating", leave=False):
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)

                correct += (pred == y).sum().item()
                total += y.size(0)

        return correct / max(1, total)

    val_acc = _accuracy(val_loader)
    test_acc = _accuracy(test_loader)

    return val_acc, {
        "test_accuracy": round(test_acc, 4),
        "compute_time_sec": round(compute_time, 3),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": len(test_dataset),
    }
