import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_DIR    = os.path.join(PROJECT_ROOT, "data", "own")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data", "augmented")
BG_DIR       = os.path.join(PROJECT_ROOT, "data", "backgrounds")

CATEGORIES = ["plastic", "paper", "glass", "bio", "mixed"]
IMG_SIZE = 128

USE_BACKGROUNDS = os.path.exists(BG_DIR) and len(os.listdir(BG_DIR)) > 0


def center_crop(img, crop_pct=0.20):
    w, h = img.size
    left = int(w * crop_pct)
    top = int(h * crop_pct)
    right = w - int(w * crop_pct)
    bottom = h - int(h * crop_pct)
    return img.crop((left, top, right, bottom)).resize((w, h), Image.LANCZOS)


def random_crop(img):
    w, h = img.size
    p = random.uniform(0.10, 0.25)
    l, t = int(w * random.uniform(0, p)), int(h * random.uniform(0, p))
    r, b = w - int(w * random.uniform(0, p)), h - int(h * random.uniform(0, p))
    return img.crop((l, t, r, b)).resize((w, h), Image.LANCZOS)


def add_noise(img):
    arr = np.asarray(img).astype("float32")
    noise = np.random.normal(0, random.uniform(5, 12), arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype("uint8"))


def color_jitter(img):
    arr = np.asarray(img).astype("float32")
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] * random.uniform(0.8, 1.2) + random.uniform(-20, 20), 0, 255)
    return Image.fromarray(arr.astype("uint8"))


def random_background(img):
    if not USE_BACKGROUNDS:
        return img
    bg_name = random.choice(os.listdir(BG_DIR))
    bg = Image.open(os.path.join(BG_DIR, bg_name)).convert("RGB")
    bg = bg.resize(img.size, Image.LANCZOS)
    gray = img.convert("L")
    mask = gray.point(lambda p: 255 if p > 200 else 0).convert("L")
    return Image.composite(img, bg, mask)


def augment_image(img):
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