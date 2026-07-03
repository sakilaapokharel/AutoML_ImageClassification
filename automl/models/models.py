"""
Model search space (CNN, ResNet18, MobileNetV2 with ImageNet weights).
Supports variable image sizes + 1/3 channel inputs.
"""

import torch
import torch.nn as nn
from torchvision import models


# -----------------------------
# Scratch CNN (no pretrained)
# -----------------------------
def head(in_features, num_classes):
    head = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),

    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(256, num_classes)
)
    return head
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

            # works for ANY image size
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# -----------------------------
# Helper: convert RGB weights -> grayscale
# -----------------------------
def adapt_first_conv(conv, in_channels):
    """
    Converts pretrained conv (RGB) → new conv (grayscale or RGB).
    Preserves pretrained knowledge.
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
            raise ValueError("Unsupported channel config")

    return new_conv


# -----------------------------
# Model builder
# -----------------------------
def build_model(name, num_classes, in_channels=3):
    name = name.lower()
    
    print(f"Training with {name}")
    
    # -------------------------
    # Scratch CNN
    # -------------------------
    if name == "scratch_cnn":
        return ScratchCNN(num_classes, in_channels)

    # -------------------------
    # ResNet18
    # -------------------------
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # adapt input channels if needed
        m.conv1 = adapt_first_conv(m.conv1, in_channels)

        # freeze backbone
        for p in m.parameters():
            p.requires_grad = False

        # replace classifier (trainable)
        m.fc = head(m.fc.in_features, num_classes)

        return m

    # -------------------------
    # MobileNetV2
    # -------------------------
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # adapt first conv layer
        first_conv = m.features[0][0]
        m.features[0][0] = adapt_first_conv(first_conv, in_channels)

        # freeze backbone
        for p in m.parameters():
            p.requires_grad = False

        # replace classifier
        m.classifier[1] = head(
            m.classifier[1].in_features,
            num_classes
        )
        
        return m

    raise ValueError(f"Unknown model '{name}'")


# -----------------------------
# Utility: trainable params
# -----------------------------
def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]