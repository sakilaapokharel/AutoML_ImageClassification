from pathlib import Path

# This file lives in automl_cnn_tabpfn/, repo root is one level up
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

DATASET_DIRS = {
    "emotions": DATA_ROOT / "emotions",
    "fashion": DATA_ROOT / "fashion",
    "flowers": DATA_ROOT / "flowers",
}

MODEL_CHOICES = ["resnet18", "efficientnet_b0", "mobilenet", "densenet121"]