# Comprehensive Disease Prediction & Triage System

An AI-driven medical diagnostic ecosystem developed to predict and classify various diseases using structured clinical data, symptom analysis, and medical imaging. This project leverages multiple Machine Learning algorithms and Deep Learning techniques to achieve high-accuracy clinical predictions.

## 🚀 Key Features & Models

### 1. General Disease Prediction (Clinical Data)
Models trained on structured patient data to predict the likelihood of severe conditions. Handling highly imbalanced datasets using SMOTE, hyperparameter tuning, and cross-validation to maximize precision and recall.
* **Diseases Covered:** Heart Disease, Diabetes, Stroke, etc.
* **Algorithms Used:** XGBoost, Random Forest, Support Vector Machines (SVM), Logistic Regression.
* **Key Techniques:** KNN Imputation, SMOTE (Synthetic Minority Over-sampling Technique), Standard Scaling.

### 2. Symptom-Based Triage System
A high-dimensional NLP-style classification model designed to map patient symptoms directly to one of 754 potential diseases.
* **Scale:** Trained on nearly 250,000 patient records across 754 unique disease categories.
* **Algorithm:** Highly optimized `Logistic Regression` and `Linear SVC`.
* **Performance:** Achieved **86.7% accuracy** (operating at ~95% of the dataset's maximum theoretical limit of 91.6% due to overlapping symptom profiles).

### 3. Brain Tumor MRI Classification (Deep Learning)
A computer vision model trained to analyze MRI scans and classify them into four categories: Glioma, Meningioma, Pituitary, or No Tumor.
* **Algorithm:** Convolutional Neural Network (CNN) utilizing **Transfer Learning** via **MobileNetV2**.
* **Techniques:** Data augmentation (rotation, flipping, zooming), multi-stage fine-tuning (unfreezing deep layers), and native 224x224 image resolution processing.
* **Performance:** Reached **89% overall test accuracy**, with exceptional recall for Pituitary (99%) and No Tumor (99%) categories.

## 🛠️ Technology Stack
* **Language:** Python 3.11.9
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
* **Deep Learning:** TensorFlow & Keras
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

## 📂 Repository Structure
* `train_all_disease.py`: Automated pipeline for cleaning, balancing, and training optimized models for standalone structured datasets (e.g. Heart Disease, Diabetes).
* `train_symptom_model.py`: High-efficiency pipeline for processing massive, sparse Boolean symptom matrices into multi-class disease predictions.
* `train_brain_tumor.py`: Deep learning script for preprocessing MRI images and running multi-stage transfer learning with MobileNetV2.
* `results/`: Contains saved `.pkl` / `.keras` model binaries, confusion matrices, and ROC curves.

## 🧠 Future Enhancements
* Development of a frontend interactive user interface (Streamlit or React) to allow direct user input of symptoms, clinical variables, and MRI uploads.
* Integration of heavier vision models (e.g., ResNet50V2) to push MRI classification accuracy past 95%.
* Jwt Authentication
