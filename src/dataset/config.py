from dataset.datasets import (
    FlowersDataset,
    EmotionsDataset,
    FashionDataset,
    SkinCancerDataset,
)

DATASETS = {
    "flowers": FlowersDataset,
    "emotions": EmotionsDataset,
    "fashion": FashionDataset,
    "skin_cancer": SkinCancerDataset,
}
