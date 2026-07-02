"""
Multi-Disease Prediction - Optimised Training Script
------------------------------------------------------
Key improvements over the basic version:
  1. Heart Disease .data file is now loaded correctly (no CSV required).
  2. Missing values are imputed (KNN for numeric, mode for categorical)
     instead of dropping rows — preserves far more data.
  3. SMOTE oversampling for imbalanced datasets (e.g. Stroke: 95 vs 5 %).
  4. Hyperparameter tuning via RandomizedSearchCV with StratifiedKFold,
     so every model gets the best parameters for each disease.
  5. Added Gradient Boosting classifier alongside LR, SVM, RF, XGBoost.
  6. ROC-curve plot per disease (all models on one chart).
  7. Per-disease confusion matrices + master comparison CSV.

Run with:  python train_all_disease.py
"""

import os
import glob
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                   # headless backend — no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==================================================================
# 1.  CONFIG
# ==================================================================
BASE_DIR    = r"D:\Disease_prediction\Dataset"
RESULTS_ROOT = r"D:\Disease_prediction\results"
RANDOM_STATE = 42
CV_FOLDS     = 5          # stratified k-fold for hyperparameter search
N_ITER       = 30         # random search iterations per model
TEST_SIZE    = 0.2

DATASETS = {
    "Heart_disease": {
        "folder": "Heart_disease",
        "file_pattern": ["*.csv", "processed.cleveland.data"],
        "header": None,             # .data file has no header row
        "col_names": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope",
            "ca", "thal", "target"
        ],
        "target_candidates": ["target"],
        "drop_cols": [],
    },
    "Diabetes": {
        "folder": "diabetes",
        "file_pattern": ["*.csv"],
        "header": 0,
        "col_names": None,
        "target_candidates": ["Outcome", "class"],
        "drop_cols": [],
    },
    "Breast_Cancer": {
        "folder": "breast_cancer",
        "file_pattern": ["*.csv"],
        "header": 0,
        "col_names": None,
        "target_candidates": ["diagnosis"],
        "drop_cols": ["id", "Unnamed: 32"],
    },
    "Stroke": {
        "folder": "Stroke_prediction",
        "file_pattern": ["*.csv"],
        "header": 0,
        "col_names": None,
        "target_candidates": ["stroke"],
        "drop_cols": ["id"],
    },
    "Liver": {
        "folder": "liver",
        "file_pattern": ["*.csv"],
        "header": 0,
        "col_names": None,
        "target_candidates": ["Dataset", "Result"],
        "drop_cols": [],
    },
    "Kidney": {
        "folder": "kidney",
        "file_pattern": ["*.csv"],
        "header": 0,
        "col_names": None,
        "target_candidates": ["classification"],
        "drop_cols": ["id"],
    },
}

# ------------------------------------------------------------------
#  Hyperparameter search spaces — much wider than defaults
# ------------------------------------------------------------------
PARAM_GRIDS = {
    "Logistic Regression": {
        "model": LogisticRegression,
        "params": {
            "C": np.logspace(-3, 3, 50),
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
            "max_iter": [2000],
        },
    },
    "SVM": {
        "model": SVC,
        "params": {
            "C": np.logspace(-2, 3, 30),
            "kernel": ["rbf", "poly"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "probability": [True],
        },
    },
    "Random Forest": {
        "model": RandomForestClassifier,
        "params": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "random_state": [RANDOM_STATE],
        },
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier,
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 0.9, 1.0],
            "min_samples_split": [2, 5, 10],
            "random_state": [RANDOM_STATE],
        },
    },
    "XGBoost": {
        "model": XGBClassifier,
        "params": {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 9],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0.5, 1, 2, 5],
            "eval_metric": ["logloss"],
            "random_state": [RANDOM_STATE],
        },
    },
}


# ==================================================================
# 2.  HELPER FUNCTIONS
# ==================================================================

def load_dataset(cfg):
    """Locate and load the data file for one disease."""
    data_dir = os.path.join(BASE_DIR, cfg["folder"])
    for pattern in cfg["file_pattern"]:
        matches = glob.glob(os.path.join(data_dir, pattern))
        if matches:
            fpath = matches[0]
            df = pd.read_csv(
                fpath,
                header=cfg.get("header", 0),
                na_values=["?", "", " ", "\\t", "\t"],
            )
            if cfg.get("col_names"):
                df.columns = cfg["col_names"]
            return df, fpath
    return None, None


def preprocess(df, target_col, drop_cols):
    """
    Smart preprocessing:
      - drop ID / junk columns
      - encode target to 0/1
      - label-encode categorical features
      - KNN impute missing values (keeps rows the old code would drop)
    Returns X (DataFrame), y (Series), label_encoders dict.
    """
    # Drop unwanted columns
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # --- Target ---
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]

    if pd.api.types.is_string_dtype(y_raw) or pd.api.types.is_object_dtype(y_raw):
        positive_values = {"m", "yes", "ckd", "1", "positive", "present"}
        y = (
            y_raw.astype(str).str.strip().str.lower()
            .apply(lambda v: 1 if v in positive_values else 0)
        )
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int)
        if y.nunique() > 2:
            y = (y > 0).astype(int)
        elif set(y.unique()) == {1, 2}:      # Liver uses 1/2
            y = (y == 1).astype(int)

    X = df.drop(columns=[target_col])

    # --- Encode categoricals with LabelEncoder (preserves info) ---
    label_encoders = {}
    for col in X.select_dtypes(include=["object", "string", "category"]).columns:
        le = LabelEncoder()
        X[col] = X[col].astype(str).str.strip().str.lower()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Force all columns to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # --- KNN Imputation (much better than dropping rows) ---
    if X.isnull().any().any():
        imputer = KNNImputer(n_neighbors=5)
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    return X, y, label_encoders


def apply_smote_if_needed(X_train, y_train, threshold=0.3):
    """
    Apply SMOTE when the minority class is less than `threshold`
    fraction of the majority class.
    """
    counts = y_train.value_counts()
    minority_ratio = counts.min() / counts.max()
    if minority_ratio < threshold:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"    SMOTE applied: {dict(counts)} -> {dict(pd.Series(y_res).value_counts())}")
        return X_res, y_res
    return X_train, y_train


def tune_and_train(model_name, model_cfg, X_train, y_train, cv):
    """
    Run RandomizedSearchCV to find the best hyperparameters,
    then return the best estimator.
    """
    base_model = model_cfg["model"]()
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=model_cfg["params"],
        n_iter=min(N_ITER, len(model_cfg["params"])**2),  # cap iterations
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    print(f"    {model_name:22s} best CV F1={search.best_score_:.3f}  params={search.best_params_}")
    return search.best_estimator_


def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "Disease"],
                yticklabels=["Healthy", "Disease"])
    plt.title(title, fontsize=11, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(roc_data, disease_name, save_path):
    plt.figure(figsize=(7, 5.5))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    for i, (name, fpr, tpr, auc_val) in enumerate(roc_data):
        plt.plot(fpr, tpr, color=colors[i % len(colors)],
                 lw=2, label=f"{name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(f"ROC Curves — {disease_name}", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ==================================================================
# 3.  MAIN LOOP
# ==================================================================
master_summary = []
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for disease_name, cfg in DATASETS.items():
    print("\n" + "=" * 70)
    print(f"  Processing: {disease_name}")
    print("=" * 70)

    results_dir = os.path.join(RESULTS_ROOT, disease_name)
    os.makedirs(results_dir, exist_ok=True)

    # --- Load ---
    df, fpath = load_dataset(cfg)
    if df is None:
        print(f"  [SKIPPED] No data file found.")
        continue
    print(f"  Loaded {fpath}  shape={df.shape}")

    # --- Identify target ---
    target_col = next(
        (c for c in cfg["target_candidates"] if c in df.columns), None
    )
    if target_col is None:
        print(f"  [SKIPPED] Target column not found in {list(df.columns)}")
        continue

    # --- Preprocess ---
    X, y, label_encoders = preprocess(df, target_col, cfg["drop_cols"])
    print(f"  After preprocessing: X={X.shape}, target distribution: {dict(y.value_counts())}")

    if X.shape[0] < 30 or y.nunique() < 2:
        print(f"  [SKIPPED] Not enough usable data.")
        continue

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # --- Scale ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    # --- SMOTE (only on training set to prevent data leakage) ---
    X_train_bal, y_train_bal = apply_smote_if_needed(X_train_scaled, y_train)

    # --- Tune & Train each model ---
    disease_results = []
    trained_models = {}
    roc_data = []

    print(f"  Tuning models ({N_ITER} random-search iters × {CV_FOLDS}-fold CV each) ...")
    for model_name, model_cfg in PARAM_GRIDS.items():
        best_model = tune_and_train(model_name, model_cfg, X_train_bal, y_train_bal, cv)
        trained_models[model_name] = best_model

        # Evaluate on held-out test set
        preds = best_model.predict(X_test_scaled)
        probs = best_model.predict_proba(X_test_scaled)[:, 1]

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec  = recall_score(y_test, preds, zero_division=0)
        f1   = f1_score(y_test, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = np.nan

        print(f"    TEST  {model_name:22s} | Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

        disease_results.append({
            "Disease": disease_name, "Model": model_name,
            "Accuracy": round(acc, 4), "Precision": round(prec, 4),
            "Recall": round(rec, 4), "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4) if not np.isnan(auc) else np.nan,
        })

        # ROC data
        if not np.isnan(auc):
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_data.append((model_name, fpr, tpr, auc))

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        plot_confusion_matrix(
            cm,
            f"{disease_name} — {model_name}",
            os.path.join(results_dir, f"cm_{model_name.replace(' ', '_')}.png"),
        )

    # --- ROC curves (all models on one plot) ---
    if roc_data:
        plot_roc_curves(roc_data, disease_name, os.path.join(results_dir, "roc_curves.png"))

    # --- Save comparison table ---
    disease_df = (
        pd.DataFrame(disease_results)
        .sort_values(by="F1-Score", ascending=False)
    )
    disease_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
    master_summary.extend(disease_results)

    # --- Save best model + scaler + metadata ---
    best_row = disease_df.iloc[0]
    best_model = trained_models[best_row["Model"]]
    joblib.dump(best_model, os.path.join(results_dir, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(results_dir, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(results_dir, "feature_columns.pkl"))
    joblib.dump(label_encoders, os.path.join(results_dir, "label_encoders.pkl"))

    print(f"\n  >>> Best model for {disease_name}: {best_row['Model']} "
          f"(F1={best_row['F1-Score']:.3f}, AUC={best_row['ROC-AUC']:.3f})")

# ==================================================================
# 4.  MASTER SUMMARY
# ==================================================================
master_df = pd.DataFrame(master_summary)
os.makedirs(RESULTS_ROOT, exist_ok=True)
master_df.to_csv(os.path.join(RESULTS_ROOT, "master_comparison_all_diseases.csv"), index=False)

print("\n" + "=" * 70)
print("  ALL DONE — Master comparison saved to:")
print(f"  {os.path.join(RESULTS_ROOT, 'master_comparison_all_diseases.csv')}")
print("=" * 70)
print(
    master_df.sort_values(["Disease", "F1-Score"], ascending=[True, False])
    .to_string(index=False)
)