import tensorflow as tf
import os

MODEL_PATH = os.path.join( "..", "..","data", "model_cnn.keras")
OUTPUT_PATH = os.path.join( "..", "..","data", "model_cnn.tflite")

model = tf.keras.models.load_model(MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to {OUTPUT_PATH}")