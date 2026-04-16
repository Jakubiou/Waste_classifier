import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

'''
This module performs data augmentation on the waste dataset. 
It takes original images and generates multiple variations (rotations, color shifts, 
noise, and background swaps) to increase dataset size and improve model generalization.
'''

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "own")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "augmented")
BG_DIR = os.path.join(PROJECT_ROOT, "data", "backgrounds")

CATEGORIES = ["plastic", "paper", "glass", "bio", "mixed"]
IMG_SIZE = 128

USE_BACKGROUNDS = os.path.exists(BG_DIR) and len(os.listdir(BG_DIR)) > 0


def center_crop(img, crop_pct=0.20):
    '''
    Crops the image from the center by a specified percentage.

    :params img: The input image to be cropped.
    :params crop_pct: Percentage of the image to remove from each side.
    :return: The center-cropped and resized image.
    '''

    w, h = img.size
    left = int(w * crop_pct)
    top = int(h * crop_pct)
    right = w - int(w * crop_pct)
    bottom = h - int(h * crop_pct)
    return img.crop((left, top, right, bottom)).resize((w, h), Image.LANCZOS)


def random_crop(img):
    '''
    Performs a random crop from the edges of the image to simulate different camera distances.

    :params img: The input image.
    :return: Randomly cropped image resized to original dimensions.
    '''

    w, h = img.size
    p = random.uniform(0.10, 0.25)
    l, t = int(w * random.uniform(0, p)), int(h * random.uniform(0, p))
    r, b = w - int(w * random.uniform(0, p)), h - int(h * random.uniform(0, p))
    return img.crop((l, t, r, b)).resize((w, h), Image.LANCZOS)


def add_noise(img):
    '''
    Adds random Gaussian noise to the image to simulate camera sensor grain.

    :params img: The input image.
    :return: Image with added digital noise.
    '''

    arr = np.asarray(img).astype("float32")
    noise = np.random.normal(0, random.uniform(5, 12), arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype("uint8"))


def color_jitter(img):
    '''
    Randomly adjusts the RGB color channels and brightness to simulate various lighting conditions.

    :params img: The input image.
    :return: Color-shifted image.
    '''

    arr = np.asarray(img).astype("float32")
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] * random.uniform(0.8, 1.2) + random.uniform(-20, 20), 0, 255)
    return Image.fromarray(arr.astype("uint8"))


def random_background(img):
    '''
    Attempts to replace the image background with a random texture from the backgrounds folder
    using a basic luminance mask.

    :params img: The input image (usually with a light/white background).
    :return: Image with a new composite background or original if backgrounds are missing.
    '''

    if not USE_BACKGROUNDS:
        return img
    bg_name = random.choice(os.listdir(BG_DIR))
    bg = Image.open(os.path.join(BG_DIR, bg_name)).convert("RGB")
    bg = bg.resize(img.size, Image.LANCZOS)
    gray = img.convert("L")
    mask = gray.point(lambda p: 255 if p > 200 else 0).convert("L")
    return Image.composite(img, bg, mask)


def augment_image(img):
    '''
    The main augmentation pipeline. Generates a list of multiple image variations
    including rotations, flips, brightness/contrast changes, and blurring.

    :params img: The original input image.
    :return: A list of PIL.Image objects containing all generated variations.
    '''

    results = []
    cropped = center_crop(img)
    results.append(cropped)

    results.append(cropped.transpose(Image.FLIP_LEFT_RIGHT))
    results.append(cropped.rotate(90, fillcolor=(128, 128, 128)))
    results.append(cropped.rotate(180, fillcolor=(128, 128, 128)))
    results.append(cropped.rotate(270, fillcolor=(128, 128, 128)))
    results.append(cropped.rotate(random.randint(5, 355), fillcolor=(128, 128, 128)))

    results.append(ImageEnhance.Brightness(cropped).enhance(random.uniform(0.6, 0.9)))
    results.append(ImageEnhance.Brightness(cropped).enhance(random.uniform(1.1, 1.4)))
    results.append(ImageEnhance.Contrast(cropped).enhance(random.uniform(0.6, 0.9)))
    results.append(ImageEnhance.Contrast(cropped).enhance(random.uniform(1.1, 1.4)))

    results.append(cropped.filter(ImageFilter.GaussianBlur(radius=0.5)))
    results.append(cropped.filter(ImageFilter.GaussianBlur(radius=1.0)))

    rc = random_crop(img)
    results.append(rc)
    results.append(rc.rotate(random.randint(-15, 15), fillcolor=(128, 128, 128)))

    results.append(add_noise(cropped))
    results.append(color_jitter(cropped))

    if USE_BACKGROUNDS:
        results.append(random_background(cropped))

    return results


def main():
    '''
    Orchestrates the augmentation process. Iterates through category folders,
    loads original images, applies the augmentation pipeline, and saves results to disk.

    :params: None
    :return: None
    '''

    total = 0
    for cat in CATEGORIES:
        in_dir = os.path.join(INPUT_DIR, cat)
        out_dir = os.path.join(OUTPUT_DIR, cat)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(in_dir):
            print(f"  {cat}: folder not found")
            continue

        files = [f for f in os.listdir(in_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        count = 0

        for fname in files:
            try:
                img = Image.open(os.path.join(in_dir, fname)).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                augmented = augment_image(img)
                stem = fname.rsplit(".", 1)[0]
                for i, aug in enumerate(augmented):
                    aug = aug.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                    aug.save(os.path.join(out_dir, f"{stem}_a{i}.jpg"), "JPEG", quality=85)
                    count += 1
            except Exception as e:
                print(f"Error with {fname}: {e}")

        print(f"  {cat}: {len(files)} originals -> {count} augmented")
        total += count

    print(f"\n  Total: {total} images")


if __name__ == "__main__":
    main()