SEARCH_SPACE = {
    "encoder": [
        "dinov2_vitb14",
        "resnet18",
        "efficientnet_b0",
        "mobilenet_v2",
        "densenet121",
    ],
    "embedding_dim": [
        # 128,
        None,
    ],
    "resize": [None, 224],
    "augmentation": [
        "none",
        "randaugment",
    ],
}

SUCCESSIVE_HALVING_FIDELITIES = [
    29,
    56,
]


SUCCESSIVE_HALVING_REDUCTION = 4
