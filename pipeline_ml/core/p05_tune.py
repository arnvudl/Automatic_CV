"""
tune.py — Modèle POTENTIEL & TRI (v5)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# CONFIG
ROOT = Path(__file__).parent.parent.parent
DATA_PATH    = ROOT / "data" / "processed" / "features.csv"
MODELS_DIR   = ROOT / "models"
REPORTS_DIR  = ROOT / "reports"
RANDOM_STATE = 42

# On ré-introduit le potentiel pour ne pas rater les futurs talents
V5_FEATURES = [
    "years_experience",
    "avg_job_duration",
    "education_level",
    "potential_score",
    "junior_potential",       # interaction is_junior × potential_score — boost juniors à fort potentiel
    "has_multiple_languages",
    "career_depth",
    "is_it"
]

def load_data():
    df = pd.read_csv(DATA_PATH)
    target = "passed_next_stage" if "passed_next_stage" in df.columns else "label"
    df = df[df[target].notna()].copy()
    return df[V5_FEATURES].fillna(0).values, df[target].astype(int).values

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # On remonte un peu le signal (C=0.1) pour mieux capturer les différences subtiles
        ('clf', LogisticRegression(C=0.1, solver='lbfgs', class_weight='balanced', random_state=RANDOM_STATE))
    ])

    print("Entraînement du modèle v5 (Potential Optimized)...")
    pipeline.fit(X_train, y_train)
    
    y_proba_train = pipeline.predict_proba(X_train)[:, 1]
    p, r, t = precision_recall_curve(y_train, y_proba_train)
    f1 = 2 * (p * r) / (p + r + 1e-9)
    best_threshold = t[np.argmax(f1[:-1])]

    print(f"Seuil de Tri Optimal : {best_threshold:.3f}")

    # Évaluation
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= best_threshold).astype(int)
    report = classification_report(y_test, y_pred_test, target_names=["Rejeté", "Invité"])
    print("\n--- RÉSULTATS v5 (Potentiel) ---")
    print(report)

    # Sauvegarde du rapport textuel
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "evaluation.txt", "w", encoding="utf-8") as f:
        f.write(f"Modèle : Logistic Regression (v5-Potential)\n")
        f.write(f"Seuil de Tri : {best_threshold:.3f}\n")
        f.write(f"Features : {V5_FEATURES}\n\n")
        f.write(report)
    print(f"Rapport d'évaluation écrit dans {REPORTS_DIR / 'evaluation.txt'}")

    # Sauvegarde des assets
    joblib.dump(pipeline.named_steps['clf'],    MODELS_DIR / "model.pkl")
    joblib.dump(pipeline.named_steps['scaler'], MODELS_DIR / "scaler.pkl")
    joblib.dump(best_threshold,                 MODELS_DIR / "threshold.pkl")
    joblib.dump(V5_FEATURES,                    MODELS_DIR / "feature_cols.pkl")

if __name__ == "__main__":
    main()
