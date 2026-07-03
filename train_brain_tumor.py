"""
Brain Tumor MRI Classification - Training Script (CNN / Transfer Learning)
-----------------------------------------------------------------------------
Classifies brain MRI scans into 4 categories: glioma, meningioma,
notumor, pituitary. Uses transfer learning (MobileNetV2) - much faster
to train than a CNN from scratch, and works well even on CPU.

Folder structure expected (yours already matches this):
    D:\\Disease_prediction\\Dataset\\tumer\\Training\\glioma\\...
    D:\\Disease_prediction\\Dataset\\tumer\\Training\\meningioma\\...
    D:\\Disease_prediction\\Dataset\\tumer\\Training\\notumor\\...
    D:\\Disease_prediction\\Dataset\\tumer\\Training\\pituitary\\...
    D:\\Disease_prediction\\Dataset\\tumer\\Testing\\<same 4 folders>

Run with: python train_brain_tumor.py

Requires: pip install tensorflow
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------------
BASE_DIR = r"D:\Disease_prediction\Dataset\tumer"
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TEST_DIR = os.path.join(BASE_DIR, "Testing")
RESULTS_DIR = r"D:\Disease_prediction\results\Brain_Tumor"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)   # Increased to native MobileNet resolution for much better accuracy
BATCH_SIZE = 32
EPOCHS_STAGE1 = 12       # Train top layers longer
EPOCHS_STAGE2 = 10       # Fine-tune longer to catch subtle meningioma patterns

# ------------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    validation_split=0.15,
    subset="training",
    seed=42,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    validation_split=0.15,
    subset="validation",
    seed=42,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

class_names = train_ds.class_names
print(f"Classes found: {class_names}")

# Preprocess for MobileNetV2 + light data augmentation on training set only
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
])

def prep_train(x, y):
    x = data_augmentation(x)
    x = preprocess_input(x)
    return x, y

def prep_eval(x, y):
    x = preprocess_input(x)
    return x, y

AUTOTUNE = tf.data.AUTOTUNE
train_ds_p = train_ds.map(prep_train).prefetch(AUTOTUNE)
val_ds_p = val_ds.map(prep_eval).prefetch(AUTOTUNE)
test_ds_p = test_ds.map(prep_eval).prefetch(AUTOTUNE)

# ------------------------------------------------------------------
# 3. BUILD MODEL (Transfer Learning)
# ------------------------------------------------------------------
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze base for stage 1

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n--- Stage 1: training top layers only ---")
history1 = model.fit(train_ds_p, validation_data=val_ds_p, epochs=EPOCHS_STAGE1)

# ------------------------------------------------------------------
# 4. FINE-TUNE (unfreeze last part of the base model, train a bit more)
# ------------------------------------------------------------------
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 50  # unfreeze last 50 layers for deeper fine-tuning
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n--- Stage 2: fine-tuning ---")
history2 = model.fit(train_ds_p, validation_data=val_ds_p, epochs=EPOCHS_STAGE2)

# ------------------------------------------------------------------
# 5. EVALUATE ON TEST SET
# ------------------------------------------------------------------
test_loss, test_acc = model.evaluate(test_ds_p)
print(f"\nTest Accuracy: {test_acc:.3f}")

y_true = np.concatenate([y.numpy() for _, y in test_ds_p], axis=0).argmax(axis=1)
y_pred_probs = model.predict(test_ds_p)
y_pred = y_pred_probs.argmax(axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Brain Tumor Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

# ------------------------------------------------------------------
# 6. PLOT TRAINING CURVES
# ------------------------------------------------------------------
acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]

plt.figure(figsize=(6, 4))
plt.plot(acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.axvline(EPOCHS_STAGE1, color="gray", linestyle="--", label="Fine-tuning starts")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training Progress")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"))
plt.close()

# ------------------------------------------------------------------
# 7. SAVE MODEL + CLASS NAMES
# ------------------------------------------------------------------
model.save(os.path.join(RESULTS_DIR, "brain_tumor_model.keras"))
with open(os.path.join(RESULTS_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f)

print(f"\nModel saved to: {os.path.join(RESULTS_DIR, 'brain_tumor_model.keras')}")
print(f"Results (plots, report) saved to: {RESULTS_DIR}")