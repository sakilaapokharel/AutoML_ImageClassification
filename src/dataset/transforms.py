from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from src.automl.config import PREPROCESS_POLICIES
from torch.utils.data import Dataset


class CLAHE:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Works for both grayscale and RGB images.
    """

    def __init__(
        self,
        clip_limit=2.0,
        tile_grid_size=(8, 8),
    ):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    def __call__(self, img):

        img = np.asarray(img)

        # Grayscale
        if img.ndim == 2:
            img = self.clahe.apply(img)

        # RGB
        else:
            lab = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2LAB,
            )

            l, a, b = cv2.split(lab)

            l = self.clahe.apply(l)

            lab = cv2.merge((l, a, b))

            img = cv2.cvtColor(
                lab,
                cv2.COLOR_LAB2RGB,
            )

        return Image.fromarray(img)


class GammaCorrection:
    """
    Fast gamma correction using a lookup table.
    gamma < 1 : brighten
    gamma > 1 : darken
    """

    def __init__(self, gamma=1.0):

        self.gamma = gamma

        if gamma == 1.0:
            self.table = None

        else:
            self.table = np.array(
                [((i / 255.0) ** gamma) * 255 for i in range(256)],
                dtype=np.uint8,
            )

    def __call__(self, img):

        if self.table is None:
            return img

        img = np.asarray(img)

        img = cv2.LUT(
            img,
            self.table,
        )

        return Image.fromarray(img)


class GaussianDenoise:
    """
    Gaussian smoothing.
    """

    def __init__(
        self,
        kernel_size=3,
        sigma=0,
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):

        img = np.asarray(img)

        img = cv2.GaussianBlur(
            img,
            (self.kernel_size, self.kernel_size),
            self.sigma,
        )

        return Image.fromarray(img)


class Sharpen:
    """
    Unsharp masking.
    """

    def __init__(
        self,
        alpha=1.5,
        kernel_size=3,
    ):
        self.alpha = alpha
        self.kernel_size = kernel_size

    def __call__(self, img):

        img = np.asarray(img)

        blurred = cv2.GaussianBlur(
            img,
            (self.kernel_size, self.kernel_size),
            0,
        )

        sharpened = cv2.addWeighted(
            img,
            self.alpha,
            blurred,
            1.0 - self.alpha,
            0,
        )

        sharpened = np.clip(
            sharpened,
            0,
            255,
        ).astype(np.uint8)

        return Image.fromarray(sharpened)


def build_transform(
    image_size=224,
    preprocess_policy="imagenet_default",
    train=True,
):
    policy = PREPROCESS_POLICIES[preprocess_policy]

    ops = []

    # ----- preprocessing -----

    if policy["clahe"]:
        ops.append(CLAHE())

    if policy["gamma"] is not None:
        ops.append(GammaCorrection(policy["gamma"]))

    if policy["denoise"]:
        ops.append(GaussianDenoise())

    if policy["sharpen"]:
        ops.append(Sharpen())

    # ----- resize -----

    ops.append(transforms.Resize((image_size, image_size)))

    # ----- augmentation (TRAIN ONLY) -----

    if train:

        if policy["augmentation"] == "weak":

            ops.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                ]
            )

        elif policy["augmentation"] == "randaugment":

            ops.append(
                transforms.RandAugment(
                    num_ops=2,
                    magnitude=5,
                )
            )

    # ----- tensor -----

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transforms.Compose(ops)


class TransformDataset(Dataset):

    def __init__(
        self,
        dataset,
        transform,
    ):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image, label = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
