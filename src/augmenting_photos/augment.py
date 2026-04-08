import os, random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_DIR    = os.path.join(PROJECT_ROOT, "data", "own")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data", "augmented")

CATEGORIES = ["plastic", "paper", "glass", "bio", "mixed"]
IMG_SIZE = 128

NEUTRAL_COLORS = [
    (255, 255, 255), (230, 230, 230), (200, 200, 200),
    (180, 180, 180), (245, 240, 235), (235, 240, 245),
    (150, 150, 150), (210, 210, 210), (220, 215, 210),
]


def center_crop(img, crop_pct=0.35):
    w, h = img.size
    left = int(w * crop_pct)
    top = int(h * crop_pct)
    right = w - int(w * crop_pct)
    bottom = h - int(h * crop_pct)
    return img.crop((left, top, right, bottom)).resize((w, h), Image.LANCZOS)


def replace_background(img, threshold=30):
    arr = np.asarray(img).astype("float32")
    h, w = arr.shape[:2]

    corners = np.concatenate([
        arr[:10, :10].reshape(-1, 3),
        arr[:10, -10:].reshape(-1, 3),
        arr[-10:, :10].reshape(-1, 3),
        arr[-10:, -10:].reshape(-1, 3),
    ])
    bg_color = corners.mean(axis=0)

    diff = np.sqrt(((arr - bg_color) ** 2).sum(axis=2))
    bg_mask = diff < threshold

    new_bg = random.choice(NEUTRAL_COLORS)
    result = arr.copy()
    result[bg_mask] = new_bg

    return Image.fromarray(result.astype("uint8"))


def hue_shift(img):
    arr = np.asarray(img).astype("float32")
    shift = random.choice([(1, 2, 0), (2, 0, 1)])
    shifted = np.stack([arr[:, :, shift[0]], arr[:, :, shift[1]], arr[:, :, shift[2]]], axis=2)
    return Image.fromarray(shifted.astype("uint8"))


def to_grayscale_rgb(img):
    gray = img.convert("L")
    return Image.merge("RGB", [gray, gray, gray])


def random_crop(img):
    w, h = img.size
    p = random.uniform(0.15, 0.35)
    l, t = int(w * random.uniform(0, p)), int(h * random.uniform(0, p))
    r, b = w - int(w * random.uniform(0, p)), h - int(h * random.uniform(0, p))
    return img.crop((l, t, r, b)).resize((w, h), Image.LANCZOS)


def add_noise(img):
    arr = np.asarray(img).astype("float32")
    noise = np.random.normal(0, random.uniform(8, 20), arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype("uint8"))


def color_jitter(img):
    arr = np.asarray(img).astype("float32")
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] * random.uniform(0.7, 1.3) + random.uniform(-25, 25), 0, 255)
    return Image.fromarray(arr.astype("uint8"))


def augment_image(img):
    results = []

    results.append(replace_background(img, threshold=35))

    cropped = center_crop(img, crop_pct=0.25)
    results.append(cropped)

    results.append(replace_background(cropped, threshold=35))

    results.append(cropped.transpose(Image.FLIP_LEFT_RIGHT))

    results.append(cropped.rotate(90, fillcolor=(200, 200, 200)))

    results.append(cropped.rotate(180))

    results.append(cropped.rotate(random.randint(15, 345), fillcolor=(200, 200, 200)))

    results.append(ImageEnhance.Brightness(cropped).enhance(random.uniform(1.2, 1.5)))

    results.append(ImageEnhance.Brightness(cropped).enhance(random.uniform(0.5, 0.8)))

    results.append(ImageEnhance.Contrast(cropped).enhance(random.uniform(1.3, 1.7)))

    results.append(hue_shift(cropped))

    results.append(to_grayscale_rgb(cropped))

    results.append(color_jitter(cropped))

    results.append(add_noise(cropped))

    results.append(cropped.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5))))

    results.append(random_crop(img))

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
            except Exception:
                pass

        print(f"  {cat}: {len(files)} originals → {count} augmented")
        total += count

    print(f"\n  Total: {total} images")


if __name__ == "__main__":
    main()