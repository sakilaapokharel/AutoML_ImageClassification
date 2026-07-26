SEARCH_SPACE = {
    "encoder": [
        "levit_128s",
        "tinyvit_5m",
        "deit_tiny",
        "mobilenetv3_small",
        "edgenext_xx_small",
    ],
    "embedding_dim": [
        None,
        128,
        64,
    ],
    "preprocess_policy": [
        "imagenet_default",
        "grayscale_enhanced",
        "contrast_enhanced",
        "texture_preserving",
        "strong_aug",
    ],
}

PREPROCESS_POLICIES = {
    "imagenet_default": {
        "clahe": False,
        "gamma": None,
        "denoise": False,
        "sharpen": False,
        "augmentation": "none",
    },
    "grayscale_enhanced": {
        "clahe": True,
        "gamma": 0.8,
        "denoise": True,
        "sharpen": False,
        "augmentation": "weak",
    },
    "contrast_enhanced": {
        "clahe": True,
        "gamma": None,
        "denoise": False,
        "sharpen": False,
        "augmentation": "weak",
    },
    "texture_preserving": {
        "clahe": False,
        "gamma": None,
        "denoise": False,
        "sharpen": True,
        "augmentation": "none",
    },
    "strong_aug": {
        "clahe": False,
        "gamma": None,
        "denoise": False,
        "sharpen": False,
        "augmentation": "randaugment",
    },
}

SUCCESSIVE_HALVING_FIDELITIES = [
    1024,
    2048,
    4096,
]


SUCCESSIVE_HALVING_REDUCTION = 4
