"""
p04_train.py — Entraînement v4 (Anti-Biais RRK/GCA)
Changements vs v3 :
  - education_level (SHAP dominant 0.529, biais académique) remplacé par
    education_adj (compressed scale : Bachelor=0.30, Master=0.70)
  - potential_per_year ajouté (GCA proxy — vitesse d'apprentissage junior)
  - Grid Search CV ajouté : optimise C, penalty, solver sur AUC-ROC (5-fold stratifié)
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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve
)

# Grille d'hyperparamètres explorée
# sklearn 1.8+ : penalty déprécié → on utilise l1_ratio avec solver=saga
#   l1_ratio=0 → L2 (Ridge)  |  l1_ratio=1 → L1 (Lasso)  |  0<r<1 → ElasticNet
PARAM_GRID = {
    "C":            [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    "l1_ratio":     [0.0, 0.5, 1.0],   # 0=L2, 0.5=ElasticNet, 1=L1
    "solver":       ["saga"],
    "class_weight": ["balanced"],
}

ROOT            = Path(__file__).parent.parent.parent
FEATURES_PATH   = ROOT / "data" / "processed" / "features.csv"
IDENTITIES_PATH = ROOT / "data" / "processed" / "identities.csv"
MODELS_DIR      = ROOT / "models"
REPORTS_DIR     = ROOT / "reports"
RANDOM_STATE    = 42

V3_FEATURES = [
    "exp_per_year_of_age", "avg_job_duration", "education_adj",
    "potential_score", "junior_potential", "has_multiple_languages",
    "career_depth", "is_it", "field_match",
    # potential_per_year retiré : inversement corrélé aux labels (senior = faible score)
    # crée une légère interaction négative avec le recall féminin
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

    # NOTE — Biais académique : tentative de sample reweighting (Bachelor boost x2.4)
    # a réduit education_adj SHAP de 0.529 → 0.158 (objectif atteint) MAIS a créé
    # une régression de fairness genre (Female recall 0.773 → 0.568, gap 1.3 → 11 pts)
    # en déplaçant le seuil adultes de 0.463 → 0.589 (trop sélectif).
    # Diagnostic : "fairness multiplicity" — corriger le biais académique (labels biaisés)
    # casse l'équité de genre précédemment obtenue. Correction impossible sans relabeling.
    # v4 revient donc à l'entraînement sans reweighting (meilleur profil global de fairness)
    # mais conserve education_adj + potential_per_year pour un feature space plus neutre.

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # ── Grid Search — optimisation sur AUC-ROC, 5-fold stratifié ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        base_model,
        PARAM_GRID,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid.fit(X_tr, y_train)

    model = grid.best_estimator_
    best_params = grid.best_params_
    best_cv_auc = grid.best_score_

    print(f"Grid Search — meilleurs params : {best_params}")
    print(f"Grid Search — AUC-ROC CV (5-fold) : {best_cv_auc:.3f}")

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
        f.write("Modele : Logistic Regression (v4-RRK-GCA)\n")
        f.write(f"Meilleurs hyperparametres : {best_params}\n")
        f.write(f"AUC-ROC CV (5-fold)       : {best_cv_auc:.3f}\n")
        f.write(f"Seuil adultes (30+) : {thr_adult:.3f}\n")
        f.write(f"Seuil juniors (<30) : {thr_junior:.3f}\n")
        f.write(f"Features : {V3_FEATURES}\n")
        f.write(f"AUC-ROC test : {auc:.3f}\n\n")
        f.write(report)

    joblib.dump(model,       MODELS_DIR / "model.pkl")
    joblib.dump(scaler,      MODELS_DIR / "scaler.pkl")
    joblib.dump(V3_FEATURES, MODELS_DIR / "feature_cols.pkl")
    joblib.dump(thr_adult,   MODELS_DIR / "threshold.pkl")
    joblib.dump(thr_junior,  MODELS_DIR / "threshold_junior.pkl")
    print(f"Modele sauvegarde dans {MODELS_DIR}")


if __name__ == "__main__":
    main()
