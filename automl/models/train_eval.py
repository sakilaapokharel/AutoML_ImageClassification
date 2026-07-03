"""Trains + evaluates a single (model, augmentation, loss) config for a given
epoch budget. This is the function Hyperband calls repeatedly at each rung.

'augmentation' search values, matching the diagram:
    - 'rand_balance': RandAugment applied to every image + WeightedRandomSampler
      to balance classes during training.
    - 'none': plain resize/tensor/normalize, no augmentation, natural sampling.

Test accuracy is measured on the dataset's real, official test set (e.g. your
test.csv/images_test) -- it is never touched by Hyperband's pruning decisions.
A small slice of the *train* set is held out as 'val' purely so Hyperband has
something to rank configs by; that val split is internal bookkeeping, not the
number that gets reported.
"""
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from automl.models.models import build_model, trainable_parameters
from automl.models.losses import get_loss
from automl.utils import get_targets, TransformedDataset

from tqdm import tqdm
import random

IMG_SIZE = 224
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_transform(aug_mode):
    ops = []

    if aug_mode == "rand_aug":
        print("Random Augumentation Applied")
        mag = random.choice([5, 7, 9, 11, 13, 15])
        ops.append(transforms.RandAugment(num_ops=2, magnitude=mag))
    
    ops.extend([
        transforms.ToTensor(),
        NORMALIZE,
    ])

    return transforms.Compose(ops)

def train_one_config(train_variant, test_dataset, num_classes, config, n_epochs,
                      device="cpu", val_split=0.15, batch_size=32, seed=0):
    """
    train_variant: a meta-augmented training-set variant (PIL images, no
        transform yet) from meta_augmentation -- gets split into train/val.
    test_dataset: the dataset's real, fixed test set (PIL images, no
        transform yet) -- always evaluated with the plain 'none' transform,
        never split or resampled, shared across every hyperband run.
    config: {"model": ..., "augmentation": "rand_balance"|"none", "loss": ...}

    The train/val split is deterministic given (seed, variant length), so
    repeated calls for the same variant+config across hyperband rungs reuse
    the exact same partition -- val comparisons stay apples-to-apples.

    Returns (val_accuracy, extra_info_dict) where extra_info_dict includes
    'test_accuracy' (the number to report) and 'compute_time_sec'.
    """
    targets = get_targets(train_variant)
    transform = get_transform(config["augmentation"])
    full = TransformedDataset(train_variant, transform)

    g = torch.Generator().manual_seed(seed)
    n = len(full)
    n_val = max(1, int(n * val_split))
    n_train = max(1, n - n_val)
    train_ds, val_ds = torch.utils.data.random_split(full, [n_train, n_val], generator=g)

    train_targets = [targets[i] for i in train_ds.indices]

    sampler = None
    shuffle = True
    if config["augmentation"] == "rand_balance":
        counts = torch.bincount(torch.tensor(train_targets), minlength=num_classes).float()
        weights = 1.0 / counts.clamp(min=1)
        sample_weights = [weights[t] for t in train_targets]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        shuffle = False  # mutually exclusive with sampler

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    test_transform = get_transform("none")  # never augment the real test set
    test_loader = DataLoader(
        TransformedDataset(test_dataset, test_transform), batch_size=batch_size, shuffle=False
    )

    model = build_model(config["model"], num_classes).to(device)
    
    device_ = next(model.parameters()).device
    print("Device: ",device_)

    class_counts_for_loss = torch.bincount(
        torch.tensor(train_targets), minlength=num_classes
    ).tolist()
    criterion = get_loss(config["loss"], class_counts=class_counts_for_loss, device=device)

    optimizer = torch.optim.Adam(trainable_parameters(model), lr=1e-3)

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
