import os
import io
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

RESULTS_DIR = r"D:\Disease_prediction\results"

# -------------------------------------------------------------------
# GLOBAL MODEL REGISTRY
# -------------------------------------------------------------------
MODELS = {
    "symptom": None,
    "tabular": {},
    "mri": None
}

# -------------------------------------------------------------------
# 1. LOAD SYMPTOM MODEL
# -------------------------------------------------------------------
def load_symptom_model():
    model_dir = os.path.join(RESULTS_DIR, "Symptom_Model")
    if os.path.exists(model_dir):
        try:
            model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
            label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
            symptom_columns = joblib.load(os.path.join(model_dir, "symptom_columns.pkl"))
            MODELS["symptom"] = {
                "model": model,
                "label_encoder": label_encoder,
                "symptom_columns": symptom_columns
            }
            print("[INFO] Symptom model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Loading symptom model: {e}")

# -------------------------------------------------------------------
# 2. LOAD TABULAR MODELS
# -------------------------------------------------------------------
def load_tabular_models():
    diseases = ["Breast_Cancer", "Diabetes", "heart_disease", "Kidney", "Liver", "Stroke"]
    for disease in diseases:
        disease_dir = os.path.join(RESULTS_DIR, disease)
        if os.path.exists(disease_dir):
            try:
                model = joblib.load(os.path.join(disease_dir, "best_model.pkl"))
                scaler = joblib.load(os.path.join(disease_dir, "scaler.pkl"))
                features = joblib.load(os.path.join(disease_dir, "feature_columns.pkl"))
                # label_encoders = joblib.load(os.path.join(disease_dir, "label_encoders.pkl"))
                MODELS["tabular"][disease] = {
                    "model": model,
                    "scaler": scaler,
                    "features": features
                }
                print(f"[INFO] Tabular model loaded: {disease}")
            except Exception as e:
                print(f"[ERROR] Loading {disease} model: {e}")

# -------------------------------------------------------------------
# 3. LOAD MRI MODEL
# -------------------------------------------------------------------
def load_mri_model():
    model_dir = os.path.join(RESULTS_DIR, "Brain_Tumor")
    model_path = os.path.join(model_dir, "brain_tumor_model.keras")
    classes_path = os.path.join(model_dir, "class_names.json")
    
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            with open(classes_path, "r") as f:
                class_names = json.load(f)
            MODELS["mri"] = {
                "model": model,
                "class_names": class_names,
                "img_size": (224, 224)
            }
            print("[INFO] MRI Brain Tumor model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Loading MRI model: {e}")

# Call loaders on startup
print("Loading all AI models into memory... This may take a few seconds.")
load_symptom_model()
load_tabular_models()
load_mri_model()
print("All models loaded. Server ready!")

# ===================================================================
# ROUTES
# ===================================================================

@app.route('/')
def home():
    """Renders the main SPA HTML page."""
    return render_template('index.html')

@app.route('/api/tabular/features/<disease_name>', methods=['GET'])
def get_tabular_features(disease_name):
    """Returns the list of required input features for a specific disease."""
    if disease_name not in MODELS["tabular"]:
        return jsonify({"error": "Disease not found"}), 404
    
    features = MODELS["tabular"][disease_name]["features"]
    return jsonify({"disease": disease_name, "features": features})

@app.route('/api/predict/symptom', methods=['POST'])
def predict_symptom():
    data = request.json
    user_symptoms = data.get("symptoms", [])
    
    if not user_symptoms or not MODELS["symptom"]:
        return jsonify({"error": "Invalid input or model not loaded."}), 400

    cfg = MODELS["symptom"]
    model = cfg["model"]
    le = cfg["label_encoder"]
    symptom_cols = cfg["symptom_columns"]
    
    # Create input vector
    input_vector = np.zeros(len(symptom_cols))
    
    # Handle both list of exact strings OR partial match
    matched = []
    for s in user_symptoms:
        s_clean = s.strip().lower()
        if s_clean in symptom_cols:
            idx = symptom_cols.index(s_clean)
            input_vector[idx] = 1
            matched.append(s_clean)
    
    if sum(input_vector) == 0:
        return jsonify({"error": "None of the provided symptoms matched the database."}), 400
        
    input_df = pd.DataFrame([input_vector], columns=symptom_cols)
    probs = model.predict_proba(input_df)[0]
    
    # Get top 3
    top_indices = np.argsort(probs)[::-1][:3]
    results = []
    for i in top_indices:
        disease_name = le.inverse_transform([i])[0]
        confidence = float(probs[i]) * 100
        results.append({
            "disease": disease_name,
            "confidence": round(confidence, 1)
        })
        
    return jsonify({
        "matched_symptoms": matched,
        "predictions": results
    })

@app.route('/api/predict/tabular', methods=['POST'])
def predict_tabular():
    data = request.json
    disease_name = data.get("disease")
    user_inputs = data.get("features", {})
    
    if disease_name not in MODELS["tabular"]:
        return jsonify({"error": "Invalid disease model."}), 400
        
    cfg = MODELS["tabular"][disease_name]
    model = cfg["model"]
    scaler = cfg["scaler"]
    features = cfg["features"]
    
    # Construct input dataframe in exactly the right order
    input_dict = {}
    for f in features:
        # Default to 0 if they missed a field, though frontend should prevent it
        val = float(user_inputs.get(f, 0.0))
        input_dict[f] = [val]
        
    input_df = pd.DataFrame(input_dict)
    
    try:
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        prediction = int(model.predict(input_scaled)[0])
        prob = model.predict_proba(input_scaled)[0]
        
        # We assume binary classification (0=No Disease, 1=Disease)
        # Handle cases where model might output more classes, grab the prob of the predicted class
        conf = float(prob[prediction]) * 100
        
        return jsonify({
            "disease": disease_name,
            "prediction": prediction,
            "prediction_text": "DISEASE PRESENT" if prediction == 1 else "NO DISEASE",
            "confidence": round(conf, 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/mri', methods=['POST'])
def predict_mri():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not MODELS["mri"]:
        return jsonify({"error": "MRI model not loaded"}), 500
        
    cfg = MODELS["mri"]
    model = cfg["model"]
    class_names = cfg["class_names"]
    img_size = cfg["img_size"]
    
    try:
        # Read image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize(img_size)
        
        # Convert to array
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        
        # Predict
        probs = model.predict(img_array, verbose=0)[0]
        pred_idx = np.argmax(probs)
        pred_class = class_names[pred_idx]
        conf = float(probs[pred_idx]) * 100
        
        return jsonify({
            "tumor_type": pred_class,
            "confidence": round(conf, 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run locally
    app.run(host='127.0.0.1', port=5000, debug=True)
