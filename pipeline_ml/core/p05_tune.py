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

IDENTITIES_PATH = ROOT / "data" / "processed" / "identities.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    target = "passed_next_stage" if "passed_next_stage" in df.columns else "label"
    df = df[df[target].notna()].copy()
    # Merge age pour la correction d'équité
    df_id = pd.read_csv(IDENTITIES_PATH)[['cv_id', 'age']]
    df = df.merge(df_id, on='cv_id', how='left')
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(30)
    df['is_junior_age'] = (df['age'] < 30).astype(int)
    return df[V5_FEATURES].fillna(0).values, df[target].astype(int).values, df['is_junior_age'].values

def _best_threshold_f1(y_true, y_proba):
    p, r, t = precision_recall_curve(y_true, y_proba)
    f1 = 2 * (p * r) / (p + r + 1e-9)
    return float(t[np.argmax(f1[:-1])])

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    X, y, is_junior = load_data()
    X_train, X_test, y_train, y_test, jr_train, jr_test = train_test_split(
        X, y, is_junior, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, solver='lbfgs', class_weight='balanced', random_state=RANDOM_STATE))
    ])

    print("Entraînement du modèle v5 (Potential Optimized)...")
    pipeline.fit(X_train, y_train)

    y_proba_train = pipeline.predict_proba(X_train)[:, 1]

    # Seuil global (adultes)
    adult_mask_train = jr_train == 0
    if adult_mask_train.sum() > 5:
        best_threshold = _best_threshold_f1(y_train[adult_mask_train], y_proba_train[adult_mask_train])
    else:
        best_threshold = _best_threshold_f1(y_train, y_proba_train)

    # Seuil junior : on optimise le recall uniquement pour ne pas discriminer les jeunes
    # On cherche le seuil qui donne recall >= 0.60 chez les juniors avec la meilleure precision
    junior_mask_train = jr_train == 1
    threshold_junior = best_threshold  # fallback
    if junior_mask_train.sum() > 5 and y_train[junior_mask_train].sum() > 1:
        p_j, r_j, t_j = precision_recall_curve(y_train[junior_mask_train], y_proba_train[junior_mask_train])
        # Parmi les seuils où recall >= 0.60, on prend celui avec la meilleure précision
        valid = (r_j[:-1] >= 0.60)
        if valid.any():
            best_idx = np.argmax(p_j[:-1][valid])
            threshold_junior = float(t_j[valid][best_idx])
        else:
            # Pas assez de signal, on prend le seuil qui maximise F1
            threshold_junior = _best_threshold_f1(y_train[junior_mask_train], y_proba_train[junior_mask_train])

    print(f"Seuil adultes  : {best_threshold:.3f}")
    print(f"Seuil juniors  : {threshold_junior:.3f}")

    # Évaluation avec seuils différenciés
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = np.where(
        jr_test == 1,
        (y_proba_test >= threshold_junior).astype(int),
        (y_proba_test >= best_threshold).astype(int),
    )
    report = classification_report(y_test, y_pred_test, target_names=["Rejeté", "Invité"])
    print("\n--- RÉSULTATS v5 (Fairness-Aware) ---")
    print(report)

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "evaluation.txt", "w", encoding="utf-8") as f:
        f.write("Modèle : Logistic Regression (v5-Fairness-Aware)\n")
        f.write(f"Seuil adultes (30+) : {best_threshold:.3f}\n")
        f.write(f"Seuil juniors (<30) : {threshold_junior:.3f}\n")
        f.write(f"Features : {V5_FEATURES}\n\n")
        f.write(report)
    print(f"Rapport d'évaluation écrit dans {REPORTS_DIR / 'evaluation.txt'}")

    joblib.dump(pipeline.named_steps['clf'],    MODELS_DIR / "model.pkl")
    joblib.dump(pipeline.named_steps['scaler'], MODELS_DIR / "scaler.pkl")
    joblib.dump(best_threshold,                 MODELS_DIR / "threshold.pkl")
    joblib.dump(threshold_junior,               MODELS_DIR / "threshold_junior.pkl")
    joblib.dump(V5_FEATURES,                    MODELS_DIR / "feature_cols.pkl")

if __name__ == "__main__":
    main()
