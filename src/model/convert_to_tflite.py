import tensorflow as tf
import os

'''
This module handles the conversion of a trained Keras model (.keras) 
into a TensorFlow Lite (.tflite) format. TFLite models are optimized for 
deployment on edge devices, mobile apps, and low-latency environments.
'''

MODEL_PATH = os.path.join( "..", "..","data", "model_cnn1.keras")
OUTPUT_PATH = os.path.join( "..", "..","data", "model_cnn1.tflite")

model = tf.keras.models.load_model(MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to {OUTPUT_PATH}")