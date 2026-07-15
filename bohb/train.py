# train.py

import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models
from bohb.datasets import load_dataset, TransformedDataset

# =====================================================
# Classification head
# =====================================================
import random

RGB_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

GRAY_NORMALIZE = transforms.Normalize(mean=[0.5], std=[0.5])

to_tensor = transforms.ToTensor()


def head(in_features, num_classes):

    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )


# =====================================================
# Scratch CNN
# =====================================================


class ScratchCNN(nn.Module):

    def __init__(self, num_classes, in_channels=3):

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.features(x)

        x = x.flatten(1)

        return self.classifier(x)


# =====================================================
# Adapt pretrained convolution
# =====================================================


def adapt_first_conv(conv, in_channels):

    if in_channels == 3:
        return conv

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=False,
    )

    with torch.no_grad():

        new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))

    return new_conv


# =====================================================
# Model builder
# =====================================================


def build_model(name, num_classes, in_channels, img_size, resize):

    name = name.lower()

    if name == "scratch_cnn":

        return ScratchCNN(num_classes, in_channels)

    if name == "resnet18":

        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        model.conv1 = adapt_first_conv(model.conv1, in_channels)

        for p in model.parameters():

            p.requires_grad = False

        model.fc = head(model.fc.in_features, num_classes)

        return model

    if name == "mobilenet_v2":

        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        model.features[0][0] = adapt_first_conv(model.features[0][0], in_channels)

        for p in model.parameters():

            p.requires_grad = False

        model.classifier[1] = head(model.classifier[1].in_features, num_classes)

        return model

    if name == "efficientnet_b0":

        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        model.features[0][0] = adapt_first_conv(model.features[0][0], in_channels)

        for p in model.parameters():

            p.requires_grad = False

        model.classifier[1] = head(model.classifier[1].in_features, num_classes)

        return model

    if name == "densenet121":

        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        model.features.conv0 = adapt_first_conv(model.features.conv0, in_channels)

        for p in model.parameters():

            p.requires_grad = False

        model.classifier = head(model.classifier.in_features, num_classes)

        return model

    raise ValueError(f"Unknown model {name}")


# =====================================================
# Dataset loader
# =====================================================


def get_loss(name):

    if name == "label_smoothing":

        return nn.CrossEntropyLoss(label_smoothing=0.1)

    return nn.CrossEntropyLoss()


# =====================================================
# Optimizer
# =====================================================


def get_optimizer(model, config):

    params = [p for p in model.parameters() if p.requires_grad]

    if config["optimizer"] == "SGD":

        return optim.SGD(
            params,
            lr=config["learning_rate"],
            momentum=0.9,
            weight_decay=config["weight_decay"],
        )

    return optim.AdamW(
        params, lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )


# =====================================================
# Train one epoch
# =====================================================


def train_epoch(model, loader, optimizer, criterion, device):

    model.train()

    pbar = tqdm(loader, desc="Training", leave=False)

    running_loss = 0.0

    for batch_idx, (x, y) in enumerate(pbar, 1):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        pbar.set_postfix(loss=f"{running_loss / batch_idx:.4f}")


# =====================================================
# Evaluation
# =====================================================

@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Validation", leave=False)

    for x, y in pbar:

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x).argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_postfix(acc=f"{correct / total:.4f}")

    return correct / total


# =====================================================
# BOHB entry point
# =====================================================
def get_transform(aug_mode, resize, in_channels=3):
    ops = []

    if resize == 1:
        print("Resize: 224x224")
        ops.append(transforms.Resize((224, 224)))

    if aug_mode == "none":
        print("No Augmentation Applied")

    elif aug_mode == "rand_aug":
        print("RandAugment Applied")
        mag = random.choice([3, 5, 7, 9])
        ops.append(transforms.RandAugment(num_ops=2, magnitude=mag))

    else:
        raise ValueError(f"Unknown augmentation: {aug_mode}")

    ops.append(transforms.ToTensor())

    if in_channels == 1:
        ops.append(GRAY_NORMALIZE)
    else:
        ops.append(RGB_NORMALIZE)

    return transforms.Compose(ops)


def train_model(config, epochs, dataset_name, data_root):

    start = time.time()

    train_ds, val_ds, test_ds, num_classes, img_size, in_channels = load_dataset(
        dataset_name,
        data_root,
        batch_size=config["batch_size"],
    )

    train_transform = get_transform(
        config.get("augmentation", "none"), config.get("resize", 0), in_channels
    )

    eval_transform = get_transform("none", config.get("resize", 0), in_channels)

    train_ds = TransformedDataset(train_ds, train_transform)

    val_ds = TransformedDataset(val_ds, eval_transform)

    test_ds = TransformedDataset(test_ds, eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        config["model"], num_classes, in_channels, img_size, config.get("resize", 0)
    )

    model.to(device)

    criterion = get_loss(config.get("loss", "cross_entropy"))

    optimizer = get_optimizer(model, config)

    best_val_accuracy = 0

    epoch_bar = tqdm(
        range(int(epochs)),
        desc=f"{config['model']} | {dataset_name}",
        unit="epoch",
    )

    for epoch in epoch_bar:

        train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_accuracy = evaluate(
            model,
            val_loader,
            device,
        )

        best_val_accuracy = max(best_val_accuracy, val_accuracy)

        epoch_bar.set_postfix(
            val_acc=f"{val_accuracy:.4f}",
            best=f"{best_val_accuracy:.4f}",
        )

    return {"val_accuracy": best_val_accuracy, "compute_time_sec": time.time() - start}
