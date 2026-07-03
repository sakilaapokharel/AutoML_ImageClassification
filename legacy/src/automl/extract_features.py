# scripts/extract_features.py
import os

from automl.feature_extractor import MetaFeatureExtractor

specs = {
    "fashion": {
        "images_train": "data/fashion/images_train",
        "train_csv": "data/fashion/train.csv",
    },

    "emotions": {
        "images_train": "data/emotions/images_train",
        "train_csv": "data/emotions/train.csv",
    },

    "flowers": {
        "images_train": "data/flowers/images_train",
        "train_csv": "data/flowers/train.csv",
    },
    # add emotions, flowers, skin_cancer here once fashion works
}

extractor = MetaFeatureExtractor()
df = extractor.extract_all(specs)

print(df)

os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/meta_features.csv", index=False)
print("\nSaved to outputs/meta_features.csv")