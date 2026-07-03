"""
Test Script - Brain Tumor CNN Model
-----------------------------------------
Loads the saved CNN model and predicts the tumor class for a single
MRI image, or for a whole folder of test images at once.

Run with: python test_brain_tumor.py
"""

import os
import json
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

RESULTS_DIR = r"D:\Disease_prediction\results\Brain_Tumor"
MODEL_PATH = os.path.join(RESULTS_DIR, "brain_tumor_model.keras")
CLASS_NAMES_PATH = os.path.join(RESULTS_DIR, "class_names.json")

IMG_SIZE = (224, 224)  # must match the size used during training


def load_brain_tumor_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names


def predict_single_image(image_path, model, class_names):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    probs = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(probs)
    predicted_class = class_names[predicted_idx]
    confidence = probs[predicted_idx] * 100

    print(f"  {os.path.basename(image_path):40s} -> {predicted_class:15s} ({confidence:.1f}%)")
    return predicted_class, confidence


if __name__ == "__main__":
    model, class_names = load_brain_tumor_model()
    print(f"Model loaded. Classes: {class_names}")

    # ----------------------------------------------------------------
    # OPTION A: Test on a few random images from your Testing folder
    # ----------------------------------------------------------------
    test_dir = r"D:\Disease_prediction\Dataset\tumer\Testing"
    print(f"\nRunning sample predictions from: {test_dir}\n")

    for class_folder in class_names:
        folder_path = os.path.join(test_dir, class_folder)
        images = glob.glob(os.path.join(folder_path, "*"))[:2]  # 2 samples per class
        print(f"-- True class: {class_folder} --")
        for img_path in images:
            predict_single_image(img_path, model, class_names)

    # ----------------------------------------------------------------
    # OPTION B: Test on ONE specific image of your choice
    # ----------------------------------------------------------------
    # my_image = r"D:\path\to\your\image.jpg"
    # predict_single_image(my_image, model, class_names)