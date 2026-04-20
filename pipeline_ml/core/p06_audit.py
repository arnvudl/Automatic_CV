"""
p06_audit.py — Audit de Biais, Equite & Explicabilite v3
Sections :
  1. Biais structurels (sampling)
  2. Equite par groupe (genre, age, pays)
  3. Importance des variables (SHAP)
  4. Analyse genre detaillee (rappel ecart avant/apres v3)
"""

import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from sklearn.metrics import recall_score, precision_score

ROOT             = Path(__file__).parent.parent.parent
FEATURES_PATH    = ROOT / "data" / "processed" / "features.csv"
IDENTITIES_PATH  = ROOT / "data" / "processed" / "identities.csv"
MODEL_PATH       = ROOT / "models" / "model.pkl"
SCALER_PATH      = ROOT / "models" / "scaler.pkl"
FEAT_COLS_PATH   = ROOT / "models" / "feature_cols.pkl"
THRESHOLD_PATH   = ROOT / "models" / "threshold.pkl"
THRESHOLD_JR_PATH= ROOT / "models" / "threshold_junior.pkl"
REPORTS_DIR      = ROOT / "reports"

TARGET_COL = "label"

COUNTRY_PREFIXES = {
    '1': 'USA/Canada', '234': 'Nigeria', '31': 'Pays-Bas',
    '33': 'France',    '351': 'Portugal', '353': 'Irlande',
    '39': 'Italie',    '48': 'Pologne',   '49': 'Allemagne', '91': 'Inde',
}


def get_country(phone):
    if not phone or not str(phone).startswith('+'):
        return "Inconnu"
    p = str(phone)[1:]
    for length in [3, 2, 1]:
        prefix = p[:length]
        if prefix in COUNTRY_PREFIXES:
            return COUNTRY_PREFIXES[prefix]
    return "Autre"


def age_group(age):
    try:
        a = float(age)
        if a < 30:  return "Jeune (<30)"
        if a <= 45: return "Adulte (30-45)"
        return "Senior (>45)"
    except (ValueError, TypeError):
        return "Inconnu"


def load_full_data():
    df = pd.read_csv(FEATURES_PATH)
    df_id = pd.read_csv(IDENTITIES_PATH)[["cv_id", "gender", "age", "phone"]]
    df = df.merge(df_id, on="cv_id", how="left")
    target = TARGET_COL if TARGET_COL in df.columns else "passed_next_stage"
    df = df[df[target].notna()].copy()
    df["age_group"] = df["age"].apply(age_group)
    df["country"]   = df["phone"].apply(get_country)
    df["age_num"]   = pd.to_numeric(df["age"], errors="coerce").fillna(30)
    return df, target


def group_stats(y_true, y_pred, label, n):
    rec  = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    return f"  {str(label):20} n={n:3}  Recall={rec:.3f}  Precision={prec:.3f}"


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    df, target = load_full_data()
    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    feature_cols  = joblib.load(FEAT_COLS_PATH)
    threshold     = joblib.load(THRESHOLD_PATH)
    threshold_jr  = joblib.load(THRESHOLD_JR_PATH) if THRESHOLD_JR_PATH.exists() else threshold

    X      = df[feature_cols].fillna(0).values.astype(float)
    X_s    = scaler.transform(X)
    y_true = df[target].values.astype(int)
    y_proba= model.predict_proba(X_s)[:, 1]
    is_jr  = (df["age_num"] < 30).values
    y_pred = np.where(is_jr,
                      (y_proba >= threshold_jr).astype(int),
                      (y_proba >= threshold).astype(int))

    lines = [
        "RAPPORT D'AUDIT FINAL — CV-Intelligence (v3)",
        "=" * 60,
        "",
    ]

    # ── 1. Biais structurels ─────────────────────────────────────
    lines += ["1. BIAIS STRUCTURELS (Sampling)", "=" * 30]
    lines.append(f"Nombre total de CV audites : {len(df)}")
    lines.append(f"Distribution Genre : {df['gender'].value_counts().to_dict()}")
    lines.append(f"Distribution Age   : {df['age_group'].value_counts().to_dict()}")
    lines.append(f"Distribution Pays  : {df['country'].value_counts().to_dict()}")
    if "Senior (>45)" not in df["age_group"].values:
        lines.append("!! ALERTE : Absence totale de profils Senior (>45) dans le dataset.")
    lines.append("")

    # ── 2. Equite par groupe ─────────────────────────────────────
    lines += ["2. ANALYSE D'EQUITE (Recall par Groupe)", "=" * 30]
    for col in ["gender", "age_group", "country"]:
        lines.append(f"-- Par {col} --")
        for val in sorted(df[col].unique().tolist()):
            mask = df[col] == val
            yt, yp = y_true[mask], y_pred[mask]
            if len(yt) < 1:
                continue
            lines.append(group_stats(yt, yp, val, len(yt)))
        lines.append("")

    # ── 3. SHAP ──────────────────────────────────────────────────
    lines += ["3. IMPORTANCE DES VARIABLES (SHAP)", "=" * 30]
    explainer  = shap.LinearExplainer(model, X_s)
    shap_vals  = explainer.shap_values(X_s)
    mean_shap  = np.abs(shap_vals).mean(axis=0)
    feat_imp   = pd.Series(mean_shap, index=feature_cols).sort_values(ascending=False)
    for feat, val in feat_imp.items():
        lines.append(f"  {feat:28} {val:.4f}")
    lines.append("")

    # ── 4. Analyse genre detaillee ───────────────────────────────
    lines += ["4. ANALYSE GENRE DETAILLEE", "=" * 30]
    lines.append("  Ecart de recall entre hommes et femmes :")
    gender_stats = {}
    for g in ["Female", "Male"]:
        mask = df["gender"] == g
        if not mask.any():
            continue
        rec  = recall_score(y_true[mask], y_pred[mask], zero_division=0)
        prec = precision_score(y_true[mask], y_pred[mask], zero_division=0)
        gender_stats[g] = rec
        lines.append(f"  {g:8}  n={mask.sum():3}  Recall={rec:.3f}  Precision={prec:.3f}")
    if "Male" in gender_stats and "Female" in gender_stats:
        gap = gender_stats["Male"] - gender_stats["Female"]
        lines.append(f"  Ecart Male - Female : {gap:+.3f} pts de recall")
        if abs(gap) <= 0.10:
            lines.append("  -> Ecart acceptable (<= 10 pts) pour un systeme de pre-filtrage.")
        else:
            lines.append("  -> ALERTE : ecart > 10 pts, correction supplementaire recommandee.")
    lines.append("")
    lines += [
        "  Rappel methodologique :",
        "  - Modele v3 : exp_per_year_of_age remplace years_experience (SHAP v5 : 0.52)",
        "  - Correction structurelle : normalise l'experience par la duree de carriere possible",
        "  - Seuil junior abaisse pour ne pas penaliser les profils < 30 ans",
        "  - Aucun attribut protege (genre, age) utilise comme feature du modele",
    ]

    audit_path = REPORTS_DIR / "audit.txt"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Audit v3 termine. Rapport : {audit_path}")


if __name__ == "__main__":
    main()
