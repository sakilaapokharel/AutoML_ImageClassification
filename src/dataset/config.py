from dataset.datasets import (
    FlowersDataset,
    EmotionsDataset,
    FashionDataset,
    SkinCancerDataset,
    SkinCancer_Test_Dataset,
)

DATASETS = {
    "flowers": FlowersDataset,
    "emotions": EmotionsDataset,
    "fashion": FashionDataset,
    "skin_cancer": SkinCancerDataset,
    "skin_cancer_test":SkinCancer_Test_Dataset
}
