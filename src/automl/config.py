SEARCH_SPACE = {
    "encoder": [
        "resnet18",
        "efficientnet_b0",
        "mobilenet_v2",
        "densenet121",
    ],
    "embedding_dim": [
        32,
        64,
        128,
        None,
    ],
    "resize": [224, None],
    "augmentation": [
        "none",
        "randaugment",
    ],
}

SUCCESSIVE_HALVING_FIDELITIES = [
    29,
    56,
    111,
    299,
    388,
]


SUCCESSIVE_HALVING_REDUCTION = 2
