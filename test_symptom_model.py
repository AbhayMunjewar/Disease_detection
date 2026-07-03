"""
Test Script - Symptom-Based Disease Model
----------------------------------------------
Loads the saved symptom model and predicts the most likely disease(s)
given a list of symptoms the user selects.

Run with: python test_symptom_model.py
"""

import os
import joblib
import pandas as pd
import numpy as np

RESULTS_DIR = r"D:\Disease_prediction\results\Symptom_Model"


def load_symptom_model():
    model = joblib.load(os.path.join(RESULTS_DIR, "best_model.pkl"))
    label_encoder = joblib.load(os.path.join(RESULTS_DIR, "label_encoder.pkl"))
    symptom_cols = joblib.load(os.path.join(RESULTS_DIR, "symptom_columns.pkl"))
    return model, label_encoder, symptom_cols


def predict_disease(selected_symptoms, model, label_encoder, symptom_cols, top_n=3):
    """
    selected_symptoms: list of symptom column names the person has
                        (must match exact names in symptom_cols)
    """
    input_vector = pd.DataFrame([np.zeros(len(symptom_cols))], columns=symptom_cols)
    for symptom in selected_symptoms:
        if symptom in input_vector.columns:
            input_vector[symptom] = 1
        else:
            print(f"  [WARNING] '{symptom}' not recognized as a tracked symptom - ignored.")

    probs = model.predict_proba(input_vector)[0]
    top_indices = np.argsort(probs)[::-1][:top_n]

    print("\nTop predictions:")
    for idx in top_indices:
        disease_name = label_encoder.inverse_transform([idx])[0]
        print(f"  {disease_name:30s}  {probs[idx] * 100:.1f}% confidence")


if __name__ == "__main__":
    model, label_encoder, symptom_cols = load_symptom_model()

    print(f"Model loaded. Tracks {len(symptom_cols)} symptoms across {len(label_encoder.classes_)} diseases.")
    print(f"\nFirst 15 tracked symptom names (for reference):")
    print(symptom_cols[:15])

    # ----------------------------------------------------------------
    # EDIT THIS LIST with symptom column names from the printout above
    # (or df.columns from your dataset) to test a sample case
    # ----------------------------------------------------------------
    test_symptoms = symptom_cols[:3]  # placeholder: picks first 3 tracked symptoms
    print(f"\nRunning test prediction with symptoms: {test_symptoms}")
    predict_disease(test_symptoms, model, label_encoder, symptom_cols)

    print("\n" + "=" * 50)
    print("To test with YOUR OWN symptoms, edit 'test_symptoms' above with")
    print("exact column names from your dataset (case-sensitive).")
    print("=" * 50)