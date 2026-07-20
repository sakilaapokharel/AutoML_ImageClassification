# config.py

from ConfigSpace import (
    ConfigurationSpace,
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

# =====================================================
# Helper
# =====================================================


def clean_categories(values):

    return [str(v) for v in values]


# =====================================================
# Full BOHB search space
# =====================================================


def get_full_search_space():

    cs = ConfigurationSpace(seed=0)

    # -----------------------------
    # Architecture
    # -----------------------------

    cs.add_hyperparameter(
        CategoricalHyperparameter(
            "model",
            clean_categories(
                [
                    "scratch_cnn",
                    "resnet18",
                    "mobilenet_v2",
                    "efficientnet_b0",
                    "densenet121",
                ]
            ),
        )
    )

    cs.add_hyperparameter(
        CategoricalHyperparameter(
            "augmentation",
            clean_categories(
                [
                    "rand_aug",
                    "none",
                ]
            ),
        )
    )

    cs.add_hyperparameter(
        CategoricalHyperparameter(
            "sampler",
            clean_categories(
                [
                    "random",
                    "weighted",
                ]
            ),
        )
    )

    cs.add_hyperparameter(
        CategoricalHyperparameter(
            "loss",
            clean_categories(
                [
                    "cross_entropy",
                    "label_smoothing",
                    "class_balanced",
                ]
            ),
        )
    )

    cs.add_hyperparameter(CategoricalHyperparameter("resize", [0, 1]))

    # -----------------------------
    # Training hyperparameters
    # -----------------------------

    cs.add_hyperparameter(
        UniformFloatHyperparameter("learning_rate", lower=1e-5, upper=1e-1, log=True)
    )

    cs.add_hyperparameter(CategoricalHyperparameter("batch_size", [16, 32, 64, 128]))

    cs.add_hyperparameter(
        UniformFloatHyperparameter("weight_decay", lower=1e-6, upper=1e-2, log=True)
    )

    cs.add_hyperparameter(
        CategoricalHyperparameter("optimizer", clean_categories(["SGD", "AdamW"]))
    )

    return cs


# =====================================================
# Portfolio BOHB search space
# =====================================================


def get_hyperparameter_search_space():

    cs = ConfigurationSpace(seed=0)

    # Only optimize training parameters.
    # Architecture comes from portfolio.

    cs.add_hyperparameter(
        UniformFloatHyperparameter("learning_rate", lower=1e-5, upper=1e-1, log=True)
    )

    cs.add_hyperparameter(CategoricalHyperparameter("batch_size", [16, 32, 64, 128]))

    cs.add_hyperparameter(
        UniformFloatHyperparameter("weight_decay", lower=1e-6, upper=1e-2, log=True)
    )

    cs.add_hyperparameter(
        CategoricalHyperparameter("optimizer", clean_categories(["SGD", "AdamW"]))
    )

    return cs
