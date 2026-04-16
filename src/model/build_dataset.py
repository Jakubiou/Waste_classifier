import os
import pandas as pd
from PIL import Image
from lib.feature_extractor import extract_features

'''
Scans augmented image folders, extracts 15 visual features from each photo 
using the shared feature_extractor module, and compiles them into a CSV dataset.
This CSV is the "iris.data" equivalent for our waste classifier.
'''

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "augmented")
OUTPUT_CSV   = os.path.join(PROJECT_ROOT, "data", "dataset.csv")

CATEGORIES = ["plastic", "paper", "glass", "bio", "mixed"]
IMG_SIZE = 128

FEATURE_NAMES = list(extract_features(Image.new("RGB", (10, 10))).keys())


def main():
    '''
    Walks through the augmented data directory, extracts features from every
    JPEG image, and exports the final feature matrix to a CSV file.

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
                feats = extract_features(img, IMG_SIZE)
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

    print(f"  DATASET: {len(df)} rows x {len(FEATURE_NAMES)} features")

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