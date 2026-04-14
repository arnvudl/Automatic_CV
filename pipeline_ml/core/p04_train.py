"""
train.py — Entraînement Standard v5 (Coherency Pack)
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==============================================================
# CONFIG
# ==============================================================
ROOT = Path(__file__).parent.parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
MODELS_DIR    = ROOT / "models"
RANDOM_STATE  = 42

# Même liste de variables que tune.py pour la cohérence v5
V5_FEATURES = [
    "years_experience",
    "avg_job_duration",
    "education_level",
    "potential_score",
    "junior_potential",
    "has_multiple_languages",
    "career_depth",
    "is_it"
]

def load_data():
    df = pd.read_csv(FEATURES_PATH)
    target = "label" if "label" in df.columns else "passed_next_stage"
    df = df[df[target].notna()].copy()
    return df, target

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    df, target = load_data()
    
    X = df[V5_FEATURES].fillna(0).values
    y = df[target].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Modèle Standard v5
    model = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    print(classification_report(y_test, y_pred, target_names=["Rejeté", "Invité"]))

    joblib.dump(model, MODELS_DIR / "model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(V5_FEATURES, MODELS_DIR / "feature_cols.pkl")
    # On garde un seuil par défaut à 0.5 pour l'entraînement standard
    joblib.dump(0.5, MODELS_DIR / "threshold.pkl")
    
    print(f"Entraînement Standard v5 terminé sur {len(df)} lignes.")
    print(f"Modèle et Scaler sauvegardés dans {MODELS_DIR}")

if __name__ == "__main__":
    main()
