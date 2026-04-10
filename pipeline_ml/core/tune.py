"""
tune.py — Optimisation (Refactored)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler

# ==============================================================
# CONFIG (Adaptée pour core/tune.py)
# ==============================================================
ROOT = Path(__file__).parent.parent.parent
DATA_PATH    = ROOT / "data" / "processed" / "dataset.csv"
MODELS_DIR   = ROOT / "models"
RANDOM_STATE = 42

FEATURE_COLS = [
    "years_experience", "avg_job_duration", "education_level", "nb_jobs",
    "nb_methods_skills", "nb_languages", "nb_certifications", "english_level",
    "has_german", "nb_technical_skills", "log_years_exp", "exp_edu_score",
    "cert_density", "multilingual_score", "method_tech_ratio", "tech_per_year",
    "career_depth", "is_it", "is_finance"
]

def load_data():
    df = pd.read_csv(DATA_PATH)
    target = "passed_next_stage" if "passed_next_stage" in df.columns else "label"
    df = df[df[target].notna()].copy()
    X = df[FEATURE_COLS].fillna(0).values
    y = df[target].astype(int).values
    return X, y

def optimize_threshold(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    params = {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear"]}
    grid = GridSearchCV(model, params, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train_s, y_train)
    
    best_model = grid.best_estimator_
    threshold = optimize_threshold(best_model, X_train_s, y_train)
    
    joblib.dump(best_model, MODELS_DIR / "model.pkl")
    joblib.dump(scaler,      MODELS_DIR / "scaler.pkl")
    joblib.dump(threshold,   MODELS_DIR / "threshold.pkl")
    print(f"Modèle optimisé et seuil ({threshold:.3f}) sauvegardés.")

if __name__ == "__main__":
    main()
