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
    "mobilenet_v3_large": (
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.DEFAULT,
        lambda m: m.classifier[0].in_features,
        lambda m: setattr(m, "classifier", nn.Identity()),
    ),
    "densenet121": (
        models.densenet121,
        models.DenseNet121_Weights.DEFAULT,
        lambda m: m.classifier.in_features,
        lambda m: setattr(m, "classifier", nn.Identity()),
    ),
}


def get_encoder(name: str, pretrained: bool = True):
    """
    Returns
    -------
    encoder : nn.Module
        Backbone with classification head removed.
    embedding_dim : int
        Size of the output embedding.
    """
    if name not in _ENCODERS:
        raise ValueError(
            f"Unknown encoder '{name}'. " f"Available: {list(_ENCODERS.keys())}"
        )

    constructor, weights, get_dim, remove_head = _ENCODERS[name]

    model = constructor(weights=weights if pretrained else None)
    embedding_dim = get_dim(model)
    remove_head(model)

    return model, embedding_dim
