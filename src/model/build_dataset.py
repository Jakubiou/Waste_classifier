import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

'''
This module handles the feature extraction process. It scans augmented 
image folders, calculates a wide range of visual metrics (color, texture, entropy), 
and compiles them into a structured CSV dataset for machine learning models.
'''

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "augmented1")
OUTPUT_CSV   = os.path.join(PROJECT_ROOT, "data", "dataset2.csv")

CATEGORIES = ["plastic", "paper", "glass", "bio", "mixed"]
IMG_SIZE = 128


def extract_features(img):
    '''
    Extracts a comprehensive set of numerical features from an image.
    Calculates metrics for color distribution, lighting, edge complexity, and texture.

    :params img: The input image object to be analyzed.
    :return: A dictionary where keys are feature names (str) and values are calculated metrics (float/int).
    '''

    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype("float32")
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    brightness = 0.299 * r + 0.587 * g + 0.114 * b

    avg_brightness = float(brightness.mean())

    contrast = float(brightness.std())

    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)

    saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-8), 0)
    avg_saturation = float(saturation.mean())

    color_uniformity = float(saturation.std())

    warm = float(((r > b + 15)).mean())

    transparency = float((brightness > 210).mean())

    dark_ratio = float((brightness < 40).mean())

    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edges).astype("float32")

    edge_density = float(edge_arr.mean())

    edge_intensity = float(edge_arr.std())

    detail = gray.filter(ImageFilter.DETAIL)
    detail_arr = np.asarray(detail).astype("float32")
    texture_roughness = float(detail_arr.std())

    emboss = gray.filter(ImageFilter.EMBOSS)
    emboss_arr = np.asarray(emboss).astype("float32")
    smoothness = float(emboss_arr.std())

    hist = np.histogram(brightness.flatten(), bins=64, range=(0, 255))[0]
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))

    ehist = np.histogram(edge_arr.flatten(), bins=32, range=(0, 255))[0]
    ehist = ehist / ehist.sum()
    ehist = ehist[ehist > 0]
    edge_entropy = float(-np.sum(ehist * np.log2(ehist)))

    channel_var = float(np.var([r.mean(), g.mean(), b.mean()]))

    highlights = float((brightness > 240).mean())

    return {
        "brightness": round(avg_brightness, 2),
        "contrast": round(contrast, 2),
        "saturation": round(avg_saturation, 4),
        "color_uniformity": round(color_uniformity, 4),
        "warm_ratio": round(warm, 4),
        "transparency": round(transparency, 4),
        "dark_ratio": round(dark_ratio, 4),
        "edge_density": round(edge_density, 2),
        "edge_intensity": round(edge_intensity, 2),
        "texture_roughness": round(texture_roughness, 2),
        "smoothness": round(smoothness, 2),
        "entropy": round(entropy, 4),
        "edge_entropy": round(edge_entropy, 4),
        "channel_variance": round(channel_var, 2),
        "highlights": round(highlights, 4),
    }


FEATURE_NAMES = list(extract_features(Image.new("RGB", (10, 10))).keys())


def main():
    '''
    Main loop that walks through the data directory, processes every JPEG image,
    and exports the final feature matrix to a CSV file. Includes basic statistics
    of the generated dataset for verification.

    :params: None
    :return: None
    '''

    rows = []
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATA_DIR, cat)
        if not os.path.exists(cat_dir):
            print(f"  {cat}: folder not found")
            continue

        files = [f for f in os.listdir(cat_dir) if f.endswith(".jpg")]
        count = 0
        for fname in files:
            try:
                img = Image.open(os.path.join(cat_dir, fname)).convert("RGB")
                feats = extract_features(img)
                feats["filename"] = fname
                feats["category"] = cat
                rows.append(feats)
                count += 1
            except Exception:
                continue
        print(f"  {cat}: {count} photos processed")

    df = pd.DataFrame(rows)
    df = df[FEATURE_NAMES + ["category", "filename"]]
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"  DATASET: {len(df)} rows × {len(FEATURE_NAMES)} features")

    print(f"\nFirst 10 rows:")
    print(df[FEATURE_NAMES + ["category"]].head(10).to_string(index=False))

    print(f"\nCategory distribution:")
    print(df["category"].value_counts().to_string())

    print(f"\nFeature statistics:")
    print(df[FEATURE_NAMES].describe().round(3).to_string())

    print(f"\nMean per category:")
    print(df.groupby("category")[FEATURE_NAMES].mean().round(3).to_string())

    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()