import os
import io
import json
import joblib
import pandas as pd
import numpy as np
import re
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if api_key:
    api_key = api_key.strip()
else:
    print("WARNING: GROQ_API_KEY not found in environment!")

groq_client = Groq(api_key=api_key)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

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
            
            # NLP Engine Setup
            clean_symptoms = [s.replace('_', ' ') for s in symptom_columns]
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(clean_symptoms)
            
            MODELS["symptom"] = {
                "model": model,
                "label_encoder": label_encoder,
                "symptom_columns": symptom_columns,
                "nlp_vectorizer": vectorizer,
                "nlp_matrix": tfidf_matrix,
                "clean_symptoms": clean_symptoms
            }
            print("[INFO] Symptom model and NLP Engine loaded successfully.")
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

@app.route('/api/chat/router', methods=['POST'])
def chat_router():
    data = request.json
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"intent": "error", "message": "Please enter a message."})

    # The list of 377 exact symptom names
    clean_symptoms = MODELS["symptom"]["clean_symptoms"] if MODELS["symptom"] else []
    
    system_prompt = f"""You are a medical AI routing engine. Your job is to classify the user's intent based on their text.
The user will either ask to check a specific disease, OR they will describe their symptoms.
You MUST reply with a strict JSON object.

RULES:
1. If the user asks to check their risk or wants an analysis for one of these diseases: Diabetes, Breast_Cancer, heart_disease, Stroke, Liver, Kidney
   Reply with: {{"intent": "tabular_form", "disease": "<DISEASE_ID>"}}
   (Use the exact ID from the list above)
   
2. If the user is describing symptoms (e.g., "I have a fever and my head hurts"):
   You must extract ALL possible medical symptoms they are experiencing. You must map their natural language to the closest exact symptom strings from the official database.
   CRITICAL TRIAGE RULES:
   - If a user says "it hurts to swallow" or "scratchy throat", DO NOT extract "difficulty in swallowing". "Difficulty in swallowing" means food gets stuck (dysphagia). Just extract "sore throat" and "throat irritation".
   - Do not over-extract severe symptoms for mild complaints.
   Official Database: {json.dumps(clean_symptoms)}
   Reply with: {{"intent": "symptom_prediction", "symptoms": ["exact symptom 1", "exact symptom 2"]}}
   Make sure every string in the "symptoms" array exactly matches a string in the Official Database.
   
3. If you absolutely cannot understand the medical context, reply with:
   {{"intent": "error", "message": "I didn't understand. Could you rephrase your medical request?"}}
"""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        response_json = json.loads(completion.choices[0].message.content)
        
        intent = response_json.get("intent")
        
        if intent == "tabular_form":
            disease_id = response_json.get("disease")
            features = MODELS["tabular"][disease_id]["features"]
            return jsonify({
                "intent": "tabular_form",
                "disease": disease_id,
                "features": features,
                "message": f"I can help check your risk for {disease_id.replace('_', ' ')}. Please provide the following medical details:"
            })
            
        elif intent == "symptom_prediction":
            extracted_symptoms = response_json.get("symptoms", [])
            
            cfg = MODELS["symptom"]
            model = cfg["model"]
            le = cfg["label_encoder"]
            symptom_cols = cfg["symptom_columns"]
            
            input_vector = np.zeros(len(symptom_cols))
            matched = []
            
            for s in extracted_symptoms:
                if s in cfg["clean_symptoms"]:
                    idx = cfg["clean_symptoms"].index(s)
                    input_vector[idx] = 1
                    matched.append(s)
            
            if sum(input_vector) == 0:
                return jsonify({
                    "intent": "error", 
                    "message": "I understood your symptoms, but I couldn't map them to my database. Please try describing them differently."
                })
                
            input_df = pd.DataFrame([input_vector], columns=symptom_cols)
            probs = model.predict_proba(input_df)[0]
            
            # Confidence Thresholding
            top_idx = np.argsort(probs)[::-1][0]
            if probs[top_idx] < 0.10:
                return jsonify({
                    "intent": "error",
                    "message": f"Based on your symptoms (<b>{', '.join(matched)}</b>), my confidence is too low to make a safe prediction. Please provide more specific symptoms (e.g. fever, fatigue, vision changes, etc.) so I can narrow it down."
                })
                
            # Filter out terrifying diseases if confidence is low
            scary_keywords = ["cancer", "tumor", "metastatic", "leukemia", "melanoma", "sarcoma", "myeloma", "hiv"]
            
            results = []
            for i in np.argsort(probs)[::-1]:
                disease_name = le.inverse_transform([i])[0]
                confidence = float(probs[i]) * 100
                
                is_scary = any(kw in disease_name.lower() for kw in scary_keywords)
                if is_scary and confidence < 30.0:
                    continue # Skip showing terrifying diseases unless we are >30% sure
                    
                results.append({
                    "disease": disease_name,
                    "confidence": round(confidence, 1)
                })
                
                if len(results) == 3:
                    break
                    
            if not results: # If all top 3 were filtered out, just grab the safest top 1
                for i in np.argsort(probs)[::-1]:
                    d_name = le.inverse_transform([i])[0]
                    if not any(kw in d_name.lower() for kw in scary_keywords):
                        results.append({"disease": d_name, "confidence": round(float(probs[i])*100, 1)})
                        break
                
            return jsonify({
                "intent": "symptom_prediction",
                "matched": matched,
                "predictions": results
            })
            
        else:
            return jsonify({"intent": "error", "message": response_json.get("message", "Unknown error occurred.")})
            
    except Exception as e:
        return jsonify({"intent": "error", "message": f"LLM Error: {str(e)}"})

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
