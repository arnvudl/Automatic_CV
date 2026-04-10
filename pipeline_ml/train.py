"""
train.py — Entraînement du modèle ML pour CV-Intelligence (TechCore)

Pipeline :
  1. Chargement de features.csv
  2. Split stratifié train/val/test (64/16/20)
  3. Scaling
  4. Entraînement : Régression Logistique, Random Forest, XGBoost
  5. Évaluation sur test set
  6. Sauvegarde du meilleur modèle + scaler

Sorties :
  models/model.pkl    — meilleur modèle
  models/scaler.pkl   — StandardScaler fitté sur train
  reports/evaluation.txt — métriques du test set
"""

import csv
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==============================================================
# CONFIG
# ==============================================================
FEATURES_PATH  = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
MODELS_DIR     = Path(__file__).parent.parent / "models"
REPORTS_DIR    = Path(__file__).parent.parent / "reports"
RANDOM_STATE   = 42

# Features passées au modèle — jamais age, gender, nom, email, téléphone
# Features engineered ajoutées (feature_engineering.py) :
#   exp_edu_score (r=+0.364), career_depth (r=+0.257), tech_per_year (r=-0.243)
FEATURE_COLS = [
    # Features brutes (corrélation > 0.10 avec label)
    "years_experience",
    "avg_job_duration",
    "education_level",
    "nb_jobs",
    "nb_methods_skills",
    "nb_languages",
    "nb_certifications",
    "english_level",
    "has_german",
    "nb_technical_skills",
    # Features engineered
    "log_years_exp",
    "exp_edu_score",
    "cert_density",
    "multilingual_score",
    "method_tech_ratio",
    "tech_per_year",
    "career_depth",
    "is_it",
    "is_finance",
]
TARGET_COL = "label"

# ==============================================================
# CHARGEMENT
# ==============================================================
def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, dtype=str)

    # Supprimer les CV sans label (pas dans student_labels.csv)
    before = len(df)
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] != "")]
    dropped = before - len(df)
    if dropped:
        print(f"  {dropped} CV sans label ignores (pas dans student_labels.csv)")

    # Colonnes numériques
    num_cols = FEATURE_COLS + [TARGET_COL]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Imputation uniquement pour les features (pas le label)
    missing_feat = df[FEATURE_COLS].isnull().sum()
    if missing_feat.any():
        print("Valeurs manquantes dans les features, imputation a 0 :")
        print(missing_feat[missing_feat > 0])
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    return df


# ==============================================================
# ÉVALUATION
# ==============================================================
def evaluate(name: str, model, X_test, y_test, scaler=None) -> dict:
    X = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Rejeté", "Invité"])

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"F1-Score : {f1:.3f}   ROC-AUC : {auc:.3f}")
    print(f"Confusion matrix :\n{cm}")
    print(report)

    return {"name": name, "f1": f1, "auc": auc, "model": model, "report": report, "cm": cm}


# ==============================================================
# MAIN
# ==============================================================
def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # --- Chargement ---
    df = load_data()
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    print(f"Dataset : {len(df)} CV  |  Invités : {y.sum()}  Rejetés : {(y==0).sum()}")

    # --- Split stratifié ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.20, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"Split  : train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # --- Scaling (fit sur train uniquement) ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # --- Entraînement ---
    models = {
        "Regression Logistique": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
            True,   # nécessite scaling
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE,
                                   class_weight="balanced", n_jobs=-1),
            False,
        ),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = (
            XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                          random_state=RANDOM_STATE, n_jobs=-1,
                          scale_pos_weight=(y == 0).sum() / y.sum(),
                          eval_metric="logloss", verbosity=0),
            False,
        )
    except ImportError:
        print("XGBoost non installé — ignoré. (pip install xgboost)")

    # Cross-validation sur train (5-fold)
    print("\n--- Cross-validation sur train (F1, 5-fold) ---")
    for name, (model, needs_scaling) in models.items():
        X_cv = X_train_s if needs_scaling else X_train
        cv_scores = cross_val_score(model, X_cv, y_train,
                                    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                                    scoring="f1", n_jobs=-1)
        print(f"  {name:<30} : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Entraînement final + évaluation test
    print("\n--- Evaluation sur test set ---")
    results = []
    for name, (model, needs_scaling) in models.items():
        X_tr = X_train_s if needs_scaling else X_train
        model.fit(X_tr, y_train)
        sc = scaler if needs_scaling else None
        results.append(evaluate(name, model, X_test, y_test, sc))

    # --- Meilleur modèle (F1) ---
    best = max(results, key=lambda r: r["f1"])
    print(f"\nMeilleur modele : {best['name']}  (F1={best['f1']:.3f}, AUC={best['auc']:.3f})")

    # --- Sauvegarde ---
    joblib.dump(best["model"], MODELS_DIR / "model.pkl")
    joblib.dump(scaler,        MODELS_DIR / "scaler.pkl")
    joblib.dump(FEATURE_COLS,  MODELS_DIR / "feature_cols.pkl")

    report_path = REPORTS_DIR / "evaluation.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Meilleur modele : {best['name']}\n")
        f.write(f"F1-Score        : {best['f1']:.3f}\n")
        f.write(f"ROC-AUC         : {best['auc']:.3f}\n\n")
        f.write("Confusion matrix :\n")
        f.write(str(best["cm"]) + "\n\n")
        f.write("Classification report :\n")
        f.write(best["report"])

    print(f"\nModele sauvegarde     : {MODELS_DIR / 'model.pkl'}")
    print(f"Rapport sauvegarde    : {report_path}")


if __name__ == "__main__":
    main()
