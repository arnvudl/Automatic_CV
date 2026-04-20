"""
p04_train.py — Entraînement v3 (Anti-Biais, Fairness-Aware)
Features : exp_per_year_of_age remplace years_experience + field_match
Seuils   : adultes (F1-optimal) et juniors (recall >= 0.55) calculés sur train
Sorties  : model.pkl, scaler.pkl, feature_cols.pkl, threshold.pkl,
           threshold_junior.pkl, reports/evaluation.txt
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve
)

ROOT            = Path(__file__).parent.parent.parent
FEATURES_PATH   = ROOT / "data" / "processed" / "features.csv"
IDENTITIES_PATH = ROOT / "data" / "processed" / "identities.csv"
MODELS_DIR      = ROOT / "models"
REPORTS_DIR     = ROOT / "reports"
RANDOM_STATE    = 42

V3_FEATURES = [
    "exp_per_year_of_age", "avg_job_duration", "education_level",
    "potential_score", "junior_potential", "has_multiple_languages",
    "career_depth", "is_it", "field_match",
]
TARGET_COL = "label"


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    df_id = pd.read_csv(IDENTITIES_PATH)[["cv_id", "age"]]
    df_id["age"] = pd.to_numeric(df_id["age"], errors="coerce").fillna(30)
    df = df.merge(df_id, on="cv_id", how="left")
    target = TARGET_COL if TARGET_COL in df.columns else "passed_next_stage"
    df = df[df[target].notna()].copy()
    df["is_junior"] = (df["age"] < 30).astype(int)
    return df, target


def best_threshold_f1(y_true, y_proba):
    p, r, t = precision_recall_curve(y_true, y_proba)
    f1 = 2 * p * r / (p + r + 1e-9)
    return float(t[np.argmax(f1[:-1])])


def best_threshold_recall(y_true, y_proba, recall_target=0.55):
    p, r, t = precision_recall_curve(y_true, y_proba)
    valid = r[:-1] >= recall_target
    if valid.any():
        return float(t[valid][np.argmax(p[:-1][valid])])
    return best_threshold_f1(y_true, y_proba)


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df, target = load_data()

    X = df[V3_FEATURES].fillna(0).values
    y = df[target].astype(int).values
    jr = df["is_junior"].values

    X_train, X_test, y_train, y_test, jr_train, jr_test = train_test_split(
        X, y, jr, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE,
                                class_weight="balanced")
    model.fit(X_tr, y_train)

    y_proba_tr = model.predict_proba(X_tr)[:, 1]
    y_proba_te = model.predict_proba(X_te)[:, 1]

    # Seuils différenciés âge
    adult_mask = jr_train == 0
    thr_adult  = best_threshold_f1(y_train[adult_mask], y_proba_tr[adult_mask])
    jr_mask    = jr_train == 1
    thr_junior = best_threshold_recall(y_train[jr_mask], y_proba_tr[jr_mask])

    y_pred = np.where(jr_test == 1,
                      (y_proba_te >= thr_junior).astype(int),
                      (y_proba_te >= thr_adult).astype(int))

    auc = roc_auc_score(y_test, y_proba_te)
    report = classification_report(y_test, y_pred, target_names=["Rejete", "Invite"])

    print(f"Modele v3 — AUC : {auc:.3f}")
    print(f"Seuil adultes (30+) : {thr_adult:.3f}  |  Seuil juniors (<30) : {thr_junior:.3f}")
    print(report)

    with open(REPORTS_DIR / "evaluation.txt", "w", encoding="utf-8") as f:
        f.write("Modele : Logistic Regression (v3-Fairness-Aware)\n")
        f.write(f"Seuil adultes (30+) : {thr_adult:.3f}\n")
        f.write(f"Seuil juniors (<30) : {thr_junior:.3f}\n")
        f.write(f"Features : {V3_FEATURES}\n")
        f.write(f"AUC-ROC  : {auc:.3f}\n\n")
        f.write(report)

    joblib.dump(model,       MODELS_DIR / "model.pkl")
    joblib.dump(scaler,      MODELS_DIR / "scaler.pkl")
    joblib.dump(V3_FEATURES, MODELS_DIR / "feature_cols.pkl")
    joblib.dump(thr_adult,   MODELS_DIR / "threshold.pkl")
    joblib.dump(thr_junior,  MODELS_DIR / "threshold_junior.pkl")
    print(f"Modele sauvegarde dans {MODELS_DIR}")


if __name__ == "__main__":
    main()
