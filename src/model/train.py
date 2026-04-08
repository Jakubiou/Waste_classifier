import os, json
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "augmented")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
CNN_PATH = os.path.join(PROJECT_ROOT, "data", "model_cnn.keras")
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
print(f"Dataset: {len(df)} rows × {len(FEATURE_NAMES)} features")
print(f"Categories: {df['category'].value_counts().to_dict()}")

label_map = {cat: i for i, cat in enumerate(CATEGORIES)}
df["label"] = df["category"].map(label_map)

X = df[FEATURE_NAMES].values.astype("float32")
y = df["label"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical



from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

images, img_labels = [], []
for idx, cat in enumerate(CATEGORIES):
    cat_dir = os.path.join(DATA_DIR, cat)
    if not os.path.exists(cat_dir):
        continue
    for fname in os.listdir(cat_dir):
        if not fname.endswith(".jpg"):
            continue
        try:
            img = Image.open(os.path.join(cat_dir, fname)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            matrix = np.asarray(img).astype("float32") / 255
            if matrix.shape == (IMG_SIZE, IMG_SIZE, 3):
                images.append(matrix)
                img_labels.append(idx)
        except Exception:
            continue
    print(f"  {cat}: {sum(1 for l in img_labels if l == idx)} images")

X_img = np.array(images)
y_img = np.array(img_labels)

Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(
    X_img, y_img, test_size=0.2, random_state=42, stratify=y_img)
yi_tr_cat = to_categorical(yi_tr, N_CLASSES)
yi_te_cat = to_categorical(yi_te, N_CLASSES)

cnn = Sequential()

cnn.add(Conv2D(32, kernel_size=(5, 5), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
cnn.add(Activation("relu"))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(64, kernel_size=(3, 3)))
cnn.add(Activation("relu"))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(64, kernel_size=(3, 3)))
cnn.add(Activation("relu"))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Flatten())
cnn.add(Dense(128))
cnn.add(Activation("relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(N_CLASSES))
cnn.add(Activation("softmax"))

cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(cnn.summary())

cnn.fit(Xi_tr, yi_tr_cat, batch_size=32, epochs=30, verbose=1,
        validation_data=(Xi_te, yi_te_cat),
        callbacks=[EarlyStopping(monitor="val_accuracy", patience=5,
                                  restore_best_weights=True)])

yi_pred = cnn.predict(Xi_te)
yi_pred_class = np.argmax(yi_pred, axis=1)
yi_test_class = np.argmax(yi_te_cat, axis=1)

acc_cnn = accuracy_score(yi_test_class, yi_pred_class)
print(f"\nCNN Accuracy: {acc_cnn*100:.1f}%")
print(classification_report(yi_test_class, yi_pred_class, target_names=CATEGORIES, digits=4))
print("Confusion matrix:")
print(confusion_matrix(yi_test_class, yi_pred_class))

cnn.save(CNN_PATH)


print(f"Conv2D CNN (raw pixels): {acc_cnn*100:.1f}%")

meta = {
    "categories": CATEGORIES,
    "category_names": CATEGORY_NAMES,
    "feature_names": FEATURE_NAMES,
    "img_size": IMG_SIZE,
    "accuracy_cnn": round(acc_cnn, 4),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "n_total": len(df),
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\nModel saved.")