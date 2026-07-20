"""
Model search space (CNN, ResNet variants, MobileNetV2, EfficientNet).
Supports variable image sizes + 1/3 channel inputs.
"""

import torch
import torch.nn as nn
from torchvision import models

# =====================================================
# Fully connected head
# =====================================================


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
# Scratch CNN (no pretrained backbone)
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
# Adapt first conv layer for 1-channel input
# =====================================================


def adapt_first_conv(conv, in_channels):
    """
    Adapt pretrained conv to different input channels.
    Supports grayscale (1 channel) or keeps RGB (3 channel).
    """
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
        if in_channels == 1:
            # RGB → grayscale: average weights
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        else:
            raise ValueError("Unsupported channel configuration")

    return new_conv


# =====================================================
# Model Builder
# =====================================================


def build_model(name, num_classes, in_channels=3, img_size=28, resize=0):
    name = name.lower()

    # -------------------------------------------------
    # Scratch CNN
    # -------------------------------------------------
    if name == "scratch_cnn":
        return ScratchCNN(num_classes, in_channels), name

    # -------------------------------------------------
    # ResNet18
    # -------------------------------------------------
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        m.conv1 = adapt_first_conv(m.conv1, in_channels)

        for p in m.parameters():
            p.requires_grad = False

        m.fc = head(m.fc.in_features, num_classes)

        return m, name

    # -------------------------------------------------
    # densenet121
    # -------------------------------------------------
    if name == "densenet121":
        if resize or img_size >= 32:
            m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

            m.features.conv0 = adapt_first_conv(m.features.conv0, in_channels)

            for p in m.parameters():
                p.requires_grad = False

            m.classifier = head(m.classifier.in_features, num_classes)

            return m, name
        else:
            print("ERROR LOADING DENSENET121 so loading scratch_cnn")
            return ScratchCNN(num_classes, in_channels), "scratch_cnn"

    # -------------------------------------------------
    # MobileNetV2 (lightweight)
    # -------------------------------------------------
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        first_conv = m.features[0][0]
        m.features[0][0] = adapt_first_conv(first_conv, in_channels)

        for p in m.parameters():
            p.requires_grad = False

        m.classifier[1] = head(m.classifier[1].in_features, num_classes)

        return m, name

    # -------------------------------------------------
    # EfficientNet-B0 (best tradeoff)
    # -------------------------------------------------
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        first_conv = m.features[0][0]
        m.features[0][0] = adapt_first_conv(first_conv, in_channels)

        for p in m.parameters():
            p.requires_grad = False

        m.classifier[1] = head(m.classifier[1].in_features, num_classes)

        return m, name

    # -------------------------------------------------
    # EfficientNet-B2 (higher capacity)
    # -------------------------------------------------
    if name == "efficientnet_b2":
        m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)

        first_conv = m.features[0][0]
        m.features[0][0] = adapt_first_conv(first_conv, in_channels)

        for p in m.parameters():
            p.requires_grad = False

        m.classifier[1] = head(m.classifier[1].in_features, num_classes)

        return m, name

    raise ValueError(f"Unknown model: {name}")


# =====================================================
# Utility: trainable parameters
# =====================================================


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
