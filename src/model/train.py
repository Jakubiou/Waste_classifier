import os, json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

'''
This module handles the training of a Convolutional Neural Network (CNN) 
for waste classification. It loads image data, balances classes, builds the model 
architecture, and evaluates performance.
'''

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR  = os.path.join(PROJECT_ROOT, "data", "augmented")
CSV_PATH  = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
CNN_PATH  = os.path.join(PROJECT_ROOT, "data", "model_cnn.keras")
META_PATH = os.path.join(PROJECT_ROOT, "data", "model_meta.json")

CATEGORIES = ["plastic", "paper", "glass", "bio", "mixed"]
CATEGORY_NAMES = {
    "plastic": "Plastic (yellow bin)",
    "paper": "Paper (blue bin)",
    "glass": "Glass (green bin)",
    "bio": "Bio (brown bin)",
    "mixed": "Mixed (black bin)",
}
N_CLASSES = len(CATEGORIES)
IMG_SIZE = 128

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
FEATURE_NAMES = [c for c in df.columns if c not in ("category", "filename")]

print("Loading images")
per_cat = {}
for idx, cat in enumerate(CATEGORIES):
    cat_dir = os.path.join(DATA_DIR, cat)
    if not os.path.exists(cat_dir):
        per_cat[cat] = []
        continue
    imgs = []
    for fname in os.listdir(cat_dir):
        if not fname.endswith(".jpg"):
            continue
        try:
            img = Image.open(os.path.join(cat_dir, fname)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            arr = np.asarray(img).astype("float32") / 255
            if arr.shape == (IMG_SIZE, IMG_SIZE, 3):
                imgs.append((arr, idx))
        except:
            continue
    per_cat[cat] = imgs
    print(f"  {cat}: {len(imgs)} images")

counts = {cat: len(per_cat[cat]) for cat in CATEGORIES}
min_count = min(c for c in counts.values() if c > 0)
print(f"\nBalancing each category to {min_count}")

all_data = []
rng = np.random.RandomState(42)
for cat in CATEGORIES:
    items = per_cat[cat]
    if len(items) == 0:
        continue
    indices = rng.choice(len(items), size=min(min_count, len(items)), replace=False)
    for i in indices:
        all_data.append(items[i])

rng.shuffle(all_data)
X = np.array([d[0] for d in all_data])
y = np.array([d[1] for d in all_data])
print(f"Total: {len(X)} images ({min_count} per category)")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
y_tr_cat = to_categorical(y_tr, N_CLASSES)
y_te_cat = to_categorical(y_te, N_CLASSES)
print(f"Train: {len(X_tr)}, Test: {len(X_te)}")

cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
class_weight = {i: w for i, w in enumerate(cw)}

cnn = Sequential()

cnn.add(Conv2D(32, kernel_size=(3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
cnn.add(BatchNormalization())
cnn.add(Activation("relu"))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(64, kernel_size=(3, 3)))
cnn.add(BatchNormalization())
cnn.add(Activation("relu"))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(128, kernel_size=(3, 3)))
cnn.add(BatchNormalization())
cnn.add(Activation("relu"))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Flatten())
cnn.add(Dense(256))
cnn.add(BatchNormalization())
cnn.add(Activation("relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(N_CLASSES))
cnn.add(Activation("softmax"))

cnn.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)
print(cnn.summary())

cnn.fit(
    X_tr, y_tr_cat,
    batch_size=32,
    epochs=5,
    verbose=1,
    validation_data=(X_te, y_te_cat),
    class_weight=class_weight,
    callbacks=[EarlyStopping(monitor="val_accuracy", patience=5,restore_best_weights=True)],
)

y_pred = cnn.predict(X_te).argmax(axis=1)
y_true = y_te_cat.argmax(axis=1)

acc = accuracy_score(y_true, y_pred)

print(f"  CNN Accuracy: {acc*100:.1f}%")

print(classification_report(y_true, y_pred, target_names=CATEGORIES, digits=3))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))

print(f"\nPredicted: {dict(zip(CATEGORIES, np.bincount(y_pred, minlength=N_CLASSES)))}")
print(f"Actual: {dict(zip(CATEGORIES, np.bincount(y_true, minlength=N_CLASSES)))}")

cnn.save(CNN_PATH)

meta = {
    "categories": CATEGORIES,
    "category_names": CATEGORY_NAMES,
    "feature_names": FEATURE_NAMES,
    "img_size": IMG_SIZE,
    "accuracy_cnn": round(acc, 4),
    "n_total": len(X),
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)