import numpy as np
import torch
import torch.nn as nn
import random
from torchvision import models
from torchvision import models, transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MIN_SAFE_SIZE = 32  # below this, DenseNet's downsampling stages fail


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resize_up_if_too_small(img):
    """Only used when resize_size=0 (original shape). Leaves images alone
    unless either dimension is below MIN_SAFE_SIZE, in which case it upscales
    just enough to be safe."""
    w, h = img.size
    if w >= MIN_SAFE_SIZE and h >= MIN_SAFE_SIZE:
        return img
    scale = MIN_SAFE_SIZE / min(w, h)
    new_w, new_h = max(MIN_SAFE_SIZE, int(w * scale)), max(
        MIN_SAFE_SIZE, int(h * scale)
    )
    return img.resize((new_w, new_h))


def build_backbone(name):
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        embed_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        embed_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
    elif name == "mobilenet":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        embed_dim = model.classifier[0].in_features
        model.classifier = nn.Identity()
    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        embed_dim = model.classifier.in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown model '{name}'")

    model.eval()
    # transform = weights.transforms()  # correct resize/normalize for this checkpoint
    return model, embed_dim


def build_transforms(resize_size=224, augment=False):
    if resize_size == 0:
        resize_step = [T.Lambda(_resize_up_if_too_small)]
    else:
        resize_step = [T.Resize((resize_size, resize_size))]

    eval_transform = T.Compose(
        resize_step
        + [
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    if not augment:
        return eval_transform, eval_transform, None

    mag = 5
    train_transform = T.Compose(
        resize_step
        + [
            T.RandAugment(num_ops=2, magnitude=mag),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform, mag


@torch.no_grad()
def extract_features(model, dataloader, device):
    model = model.to(device).eval()
    feats, labels = [], []
    for imgs, y in dataloader:
        imgs = imgs.to(device)
        out = model(imgs)
        feats.append(out.cpu().numpy())
        labels.extend(y)  # y arrives as a list of raw label values per batch
    return np.concatenate(feats, axis=0), np.array(labels)


@torch.no_grad()
def extract_features_multi_pass(model, dataset, device, batch_size):
    from torch.utils.data import DataLoader

    all_feats, all_labels = [], []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats, labels = extract_features(model, loader, device)
    all_feats.append(feats)
    all_labels.append(labels)

    return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)
