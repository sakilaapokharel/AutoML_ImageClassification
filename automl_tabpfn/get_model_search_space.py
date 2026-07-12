from itertools import product
import pandas as pd


MODELS = [
    "scratch_cnn",
    "resnet18",
    "mobilenet_v2",
    "efficientnet_b0",
    "densenet121",
]

SAMPLERS = [
    "random",
    "weighted",
]

RESIZES = [
    0,
    1,
]

AUGMENTATIONS = [
    "rand_aug",
    "none",
]

LOSSES = [
    "cross_entropy",
    "label_smoothing",
    "class_balanced",
]

EPOCHS_TRAINED = [1, 4, 11]


def generate_test_configurations(dataset_features):

    configs = []

    for model, sampler, resize, augmentation, loss, epochs in product(
        MODELS,
        SAMPLERS,
        RESIZES,
        AUGMENTATIONS,
        LOSSES,
        EPOCHS_TRAINED,
    ):

        row = dataset_features.copy()

        row.update(
            {
                "model": model,
                "sampler": sampler,
                "resize": resize,
                "augmentation": augmentation,
                "loss": loss,
                "epochs_trained": epochs,
            }
        )

        configs.append(row)

    return pd.DataFrame(configs)