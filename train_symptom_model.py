"""
Multi-Disease Symptom Prediction - Training Script
------------------------------------------------------
Trains a multi-class classifier that takes a set of symptoms (binary
columns) and predicts the most likely disease. This becomes the
"front door" of your chatbot - it does general triage, then the
chatbot can route to a specialized model (heart/diabetes/etc.) for
a more detailed follow-up prediction.

Dataset:
    D:\\Disease_prediction\\Dataset\\Disease_caution\\Final_Augmented_dataset_Diseases_and_Symptoms.csv

Run with: python train_symptom_model.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ------------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------------
DATA_PATH = r"D:\Disease_prediction\Dataset\Disease_caution\Final_Augmented_dataset_Diseases_and_Symptoms.csv"
RESULTS_DIR = r"D:\Disease_prediction\results\Symptom_Model"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------------
print(f"Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Columns (first 10): {list(df.columns[:10])} ...")

# ------------------------------------------------------------------
# 3. FIND THE TARGET (DISEASE) COLUMN
#    Common names: 'diseases', 'disease', 'prognosis'
# ------------------------------------------------------------------
target_candidates = ["diseases", "disease", "prognosis", "Disease"]
target_col = next((c for c in target_candidates if c in df.columns), None)

if target_col is None:
    # fallback: assume it's the first column (common in this dataset format)
    target_col = df.columns[0]
    print(f"Could not match known target names, defaulting to first column: '{target_col}'")
else:
    print(f"Using target column: '{target_col}'")

print(f"\nNumber of unique diseases: {df[target_col].nunique()}")
print(df[target_col].value_counts().head(10))

# ------------------------------------------------------------------
# 4. PREPROCESS
# ------------------------------------------------------------------
df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y_text = df[target_col]

# Symptom columns should already be binary (0/1). Force numeric and cast to int8 to save huge amounts of RAM!
X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.int8)

# Encode disease names into numeric labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

# Drop diseases with too few samples to be split into train/test reliably
class_counts = pd.Series(y).value_counts()
valid_classes = class_counts[class_counts >= 2].index
mask = pd.Series(y).isin(valid_classes)
X, y = X[mask.values], y[mask.values]

print(f"\nAfter filtering rare classes: {X.shape[0]} rows, {len(set(y))} disease classes remain")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 5. TRAIN MODELS
# ------------------------------------------------------------------
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

models = {
    # High accuracy, very fast for high-dimensional text/symptom data
    "Linear SVC": LinearSVC(random_state=42, dual=False, max_iter=2000),
    # Another highly optimized baseline
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
    # The fast 85% baseline you just ran
    "Naive Bayes": BernoulliNB(),
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    print(f"  Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1})
    trained_models[name] = model

# ------------------------------------------------------------------
# 6. COMPARISON TABLE
# ------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
print("\n=== MODEL COMPARISON (Symptom -> Disease) ===")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

# ------------------------------------------------------------------
# 7. SAVE BEST MODEL + LABEL ENCODER + FEATURE (SYMPTOM) LIST
#    The chatbot needs all three: model to predict, encoder to turn
#    predictions back into disease names, and the symptom list so it
#    knows what questions to ask the user.
# ------------------------------------------------------------------
best_row = results_df.iloc[0]
best_model = trained_models[best_row["Model"]]

joblib.dump(best_model, os.path.join(RESULTS_DIR, "best_model.pkl"))
joblib.dump(label_encoder, os.path.join(RESULTS_DIR, "label_encoder.pkl"))
joblib.dump(list(X.columns), os.path.join(RESULTS_DIR, "symptom_columns.pkl"))

print(f"\nBest model: {best_row['Model']} (F1={best_row['F1-Score']:.3f})")
print(f"Saved to: {RESULTS_DIR}")
print(f"Total symptoms tracked: {len(X.columns)}")
print(f"Total diseases covered: {len(label_encoder.classes_)}")