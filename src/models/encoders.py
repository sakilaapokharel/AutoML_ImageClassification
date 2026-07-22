import torch

from torch import nn
from torchvision import models

_ENCODERS = {
    "resnet18": (
        models.resnet18,
        models.ResNet18_Weights.DEFAULT,
        lambda m: m.fc.in_features,
        lambda m: setattr(m, "fc", nn.Identity()),
    ),
    "efficientnet_b0": (
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.DEFAULT,
        lambda m: m.classifier[1].in_features,
        lambda m: setattr(m, "classifier", nn.Identity()),
    ),
    "mobilenet_v2": (
        models.mobilenet_v2,
        models.MobileNet_V2_Weights.DEFAULT,
        lambda m: m.classifier[1].in_features,
        lambda m: setattr(m, "classifier", nn.Identity()),
    ),
    "densenet121": (
        models.densenet121,
        models.DenseNet121_Weights.DEFAULT,
        lambda m: m.classifier.in_features,
        lambda m: setattr(m, "classifier", nn.Identity()),
    ),
}


# ------------------------------------------------------------------
# DINOv2 (torch.hub, facebookresearch/dinov2)
# ------------------------------------------------------------------
#
# DINOv2 doesn't fit the torchvision pattern above:
#   - loaded via torch.hub, not torchvision.models
#   - forward(x) already returns a pooled [B, embedding_dim] CLS embedding
#     directly -- there's no classification head to remove
#   - embedding_dim is fixed per variant, not read off a submodule
#
# Variants with "_reg" use register tokens (recommended by the DINOv2 authors
# to reduce attention artifacts); embedding_dim is unchanged by registers.
#
# Note: patch size is 14 for all variants. torchvision's Resize in your
# transform config should ideally be a multiple of 14 (e.g. 224, 336, 518)
# for a clean patch grid -- non-multiples still run (Conv2d silently drops
# the remainder pixels) but waste a bit of the image.

_DINOV2_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14_reg": 1536,
}


def _get_dinov2_encoder(name: str, pretrained: bool):

    if not pretrained:

        raise ValueError(
            f"'{name}' is only available pretrained (loaded via torch.hub from "
            f"facebookresearch/dinov2). Random-init DINOv2 isn't supported here "
            f"-- pass pretrained=True, or use a torchvision encoder instead."
        )

    model = torch.hub.load("facebookresearch/dinov2", name)

    embedding_dim = _DINOV2_EMBED_DIMS[name]

    return model, embedding_dim


def get_encoder(name: str, pretrained: bool = True):
    """
    Returns
    -------
    encoder : nn.Module
        Backbone with classification head removed (torchvision encoders),
        or a DINOv2 backbone whose forward() already returns pooled embeddings.
    embedding_dim : int
        Size of the output embedding.
    """

    if name in _DINOV2_EMBED_DIMS:

        return _get_dinov2_encoder(name, pretrained)

    if name not in _ENCODERS:
        raise ValueError(
            f"Unknown encoder '{name}'. "
            f"Available: {list(_ENCODERS.keys()) + list(_DINOV2_EMBED_DIMS.keys())}"
        )

    constructor, weights, get_dim, remove_head = _ENCODERS[name]

    model = constructor(weights=weights if pretrained else None)
    embedding_dim = get_dim(model)
    remove_head(model)

    return model, embedding_dim