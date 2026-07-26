import timm

_ENCODERS = {
    # ---------------------------
    # CNNs
    # ---------------------------
    "resnet18": (
        "resnet18.a1_in1k",
        512,
    ),
    "edgenext_xx_small": (
        "edgenext_xx_small.in1k",
        304,
    ),
    "mobilenetv3_small": (
        "mobilenetv3_small_100.lamb_in1k",
        576,
    ),
    # ---------------------------
    # Vision Transformers
    # ---------------------------
    "deit_tiny": (
        "deit_tiny_patch16_224.fb_in1k",
        192,
    ),
    "tinyvit_5m": (
        "tiny_vit_5m_224.dist_in22k_ft_in1k",
        320,
    ),
    "levit_128s": (
        "levit_128s.fb_dist_in1k",
        384,
    ),
}


def get_encoder(name: str, pretrained: bool = True):
    """
    Returns
    -------
    encoder : nn.Module
        Backbone with classification head removed.
    embedding_dim : int
        Output embedding dimension.
    """

    if name not in _ENCODERS:
        raise ValueError(
            f"Unknown encoder '{name}'. " f"Available: {list(_ENCODERS.keys())}"
        )

    model_name, embedding_dim = _ENCODERS[name]

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,  # removes classifier head
    )

    return model, embedding_dim
