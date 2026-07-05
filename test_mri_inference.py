import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model_path = r"d:\Disease_prediction\results\Brain_Tumor\brain_tumor_model.keras"
model = load_model(model_path)
with open(r"d:\Disease_prediction\results\Brain_Tumor\class_names.json", "r") as f:
    class_names = json.load(f)

img_path = r"d:\Disease_prediction\Dataset\tumer\Testing\meningioma\Te-me_10.jpg"
if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
else:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    
    # Method 1: My manual scaling
    img_array1 = np.array(img, dtype=np.float32)
    img_array1 = (img_array1 / 127.5) - 1.0
    img_array1 = np.expand_dims(img_array1, axis=0)
    
    # Method 2: Keras preprocess
    img_array2 = np.array(img, dtype=np.float32)
    img_array2 = np.expand_dims(img_array2, axis=0)
    img_array2 = preprocess_input(img_array2)
    
    # Method 3: tf.keras.utils.image_dataset_from_directory logic
    img_tf = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array3 = tf.keras.utils.img_to_array(img_tf)
    img_array3 = np.expand_dims(img_array3, axis=0)
    img_array3 = preprocess_input(img_array3)
    
    print("Method 1 (Manual math) Prediction:")
    preds1 = model.predict(img_array1, verbose=0)[0]
    for i, name in enumerate(class_names):
        print(f"{name}: {preds1[i]*100:.2f}%")
        
    print("\nMethod 2 (preprocess_input) Prediction:")
    preds2 = model.predict(img_array2, verbose=0)[0]
    for i, name in enumerate(class_names):
        print(f"{name}: {preds2[i]*100:.2f}%")

    print("\nMethod 3 (tf utils) Prediction:")
    preds3 = model.predict(img_array3, verbose=0)[0]
    for i, name in enumerate(class_names):
        print(f"{name}: {preds3[i]*100:.2f}%")
