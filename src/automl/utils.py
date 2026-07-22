import torchvision.transforms as transforms
from PIL import Image


def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_step(text):
    print(f"\n[{text}]")


def print_progress(current, total, every=100):
    if current % every == 0 or current == total:
        print(f"## {current}/{total}")



class ResizeToMultipleOf14:

    def __call__(self, image):

        width, height = image.size

        new_width = round(width / 14) * 14
        new_height = round(height / 14) * 14

        return image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )