SEARCH_SPACE = {
    "encoder": [
        "resnet18",
        "efficientnet_b0",
        "mobilenet_v2",
        "densenet121",
    ],
    "embedding_dim": [
        # 128,
        None,
    ],
    "resize": [224, None],
    "augmentation": [
        "none",
        "randaugment",
    ],
}

SUCCESSIVE_HALVING_FIDELITIES = [
    11,
    # 12,
    # 56,
    # 111,
    # 299,
]


SUCCESSIVE_HALVING_REDUCTION = 8
