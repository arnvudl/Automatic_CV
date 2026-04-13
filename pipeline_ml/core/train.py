"""
train.py — Entraînement du modèle (Refactored)
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================================================
# CONFIG (Adaptée pour core/train.py)
# ==============================================================
ROOT = Path(__file__).parent.parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "dataset.csv"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"
RANDOM_STATE  = 42

FEATURE_COLS = [
    "exp_edu_score", "tech_per_year", "years_experience", 
    "multilingual_score", "is_it", "education_level", "avg_job_duration"
]
TARGET_COL = "passed_next_stage"

def load_data():
    global TARGET_COL
    df = pd.read_csv(FEATURES_PATH, dtype=str)
    target = TARGET_COL
    if target not in df.columns and "label" in df.columns:
        target = "label"
    df = df[df[target].notna() & (df[target] != "")]
    df[FEATURE_COLS + [target]] = df[FEATURE_COLS + [target]].apply(pd.to_numeric, errors="coerce")
    TARGET_COL = target
    return df.fillna(0)

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    df = load_data()
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    model.fit(X_train_s, y_train)

    joblib.dump(model, MODELS_DIR / "model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(FEATURE_COLS, MODELS_DIR / "feature_cols.pkl")
    print(f"Modèle sauvegardé dans {MODELS_DIR}")

if __name__ == "__main__":
    main()
