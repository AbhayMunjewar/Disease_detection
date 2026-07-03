"""
Test Script - Tabular Disease Models
----------------------------------------
Loads the saved model + scaler + feature list for each disease and
runs a prediction on a sample input (either a manually entered row,
or a random row pulled from the original dataset for a quick sanity check).

Run with: python test_tabular_models.py
"""

import os
import glob
import joblib
import pandas as pd
import numpy as np

BASE_DIR = r"D:\Disease_prediction\Dataset"
RESULTS_ROOT = r"D:\Disease_prediction\results"

DISEASE_FOLDERS = {
    "Heart_disease": "Heart_disease",
    "Diabetes": "diabetes",
    "Breast_Cancer": "breast_cancer",
    "Stroke": "Stroke_prediction",
    "Liver": "liver",
    "Kidney": "kidney",
}


def load_model_bundle(disease_name):
    """Loads model, scaler, and feature column list for one disease."""
    results_dir = os.path.join(RESULTS_ROOT, disease_name)
    model_path = os.path.join(results_dir, "best_model.pkl")
    scaler_path = os.path.join(results_dir, "scaler.pkl")
    features_path = os.path.join(results_dir, "feature_columns.pkl")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print(f"  [SKIPPED] No saved model found for {disease_name}. Train it first.")
        return None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(features_path)
    return model, scaler, feature_cols


def test_on_random_sample(disease_name, folder_name):
    """Pulls one random row from the original CSV and predicts on it -
    a quick sanity check that the saved model loads and runs correctly."""
    print(f"\n{'=' * 50}")
    print(f"Testing: {disease_name}")
    print("=" * 50)

    bundle = load_model_bundle(disease_name)
    if bundle is None:
        return
    model, scaler, feature_cols = bundle

    data_dir = os.path.join(BASE_DIR, folder_name)
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"  [SKIPPED] No CSV found to pull a sample from.")
        return

    df = pd.read_csv(csv_files[0])
    
    # Drop columns that are completely empty (like the common 'Unnamed: 32' artifact)
    df = df.dropna(axis=1, how="all")
    df = df.replace("?", np.nan).dropna()
    
    if len(df) == 0:
        print(f"  [SKIPPED] Cannot test randomly, dataset becomes empty after dropping missing values.")
        return

    # Rebuild the same one-hot encoding used in training, then align columns
    df_encoded = pd.get_dummies(df, drop_first=True)
    # Keep only feature columns the model was trained on; fill any missing with 0
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    sample = df_encoded[feature_cols].sample(1, random_state=None)

    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]
    prob = model.predict_proba(sample_scaled)[0]

    label = "DISEASE PRESENT" if pred == 1 else "NO DISEASE"
    confidence = prob[pred] * 100

    print(f"  Sample input (first 5 features): {sample.iloc[0, :5].to_dict()}")
    print(f"  Prediction: {label}")
    print(f"  Confidence: {confidence:.1f}%")


if __name__ == "__main__":
    for disease_name, folder_name in DISEASE_FOLDERS.items():
        test_on_random_sample(disease_name, folder_name)

    print("\n" + "=" * 50)
    print("Done. To test with YOUR OWN custom input instead of a random")
    print("sample, see the 'manual_predict()' pattern in the comments below.")
    print("=" * 50)

    # ----------------------------------------------------------------
    # MANUAL PREDICTION EXAMPLE (edit values, then uncomment to use)
    # ----------------------------------------------------------------
    # bundle = load_model_bundle("Diabetes")
    # if bundle:
    #     model, scaler, feature_cols = bundle
    #     my_input = {
    #         "Pregnancies": 2, "Glucose": 130, "BloodPressure": 70,
    #         "SkinThickness": 20, "Insulin": 85, "BMI": 28.5,
    #         "DiabetesPedigreeFunction": 0.5, "Age": 33
    #     }
    #     row = pd.DataFrame([my_input])[feature_cols]
    #     row_scaled = scaler.transform(row)
    #     pred = model.predict(row_scaled)[0]
    #     print("Prediction:", "Diabetes" if pred == 1 else "No Diabetes")