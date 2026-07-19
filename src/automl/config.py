SEARCH_SPACE = {
    "encoder": [
        # "resnet18",
        "efficientnet_b0",
        "mobilenet_v2",
        # "densenet121",
    ],
    "embedding_dim": [
        32,
        # 64,
        # 128,
        # 256,
        None,
    ],
    "resize": [
        224,
    ],
    "augmentation": [
        "none",
        "randaugment",
    ],
}
