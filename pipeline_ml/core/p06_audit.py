"""
audit.py — Audit de Biais & Explicabilité (v5-Final)
"""

import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from sklearn.metrics import recall_score, precision_score

# ==============================================================
# CONFIG
# ==============================================================
ROOT = Path(__file__).parent.parent.parent
FEATURES_PATH   = ROOT / "data" / "processed" / "features.csv"
IDENTITIES_PATH = ROOT / "data" / "processed" / "identities.csv"
MODEL_PATH      = ROOT / "models" / "model.pkl"
SCALER_PATH     = ROOT / "models" / "scaler.pkl"
FEAT_COLS_PATH  = ROOT / "models" / "feature_cols.pkl"
THRESHOLD_PATH  = ROOT / "models" / "threshold.pkl"
REPORTS_DIR     = ROOT / "reports"

TARGET_COL = "label"

COUNTRY_PREFIXES = {
    '1': 'USA/Canada',
    '234': 'Nigeria',
    '31': 'Pays-Bas',
    '33': 'France',
    '351': 'Portugal',
    '353': 'Irlande',
    '39': 'Italie',
    '48': 'Pologne',
    '49': 'Allemagne',
    '91': 'Inde',
}

def get_country(phone):
    if not phone or not str(phone).startswith('+'):
        return "Inconnu"
    p = str(phone)[1:] # Enlever le +
    # Essayer les préfixes de 3 chiffres, puis 2, puis 1
    for length in [3, 2, 1]:
        prefix = p[:length]
        if prefix in COUNTRY_PREFIXES:
            return COUNTRY_PREFIXES[prefix]
    return "Autre"

def age_group(age):
    try:
        a = float(age)
        if a < 30: return "Jeune (<30)"
        if a <= 45: return "Adulte (30-45)"
        return "Senior (>45)"
    except (ValueError, TypeError):
        return "Inconnu"

def load_full_data():
    df_feat = pd.read_csv(FEATURES_PATH)
    df_id = pd.read_csv(IDENTITIES_PATH)[['cv_id', 'gender', 'age', 'phone']]
    
    # Fusion sur cv_id
    df = df_feat.merge(df_id, on='cv_id', how='left')
    
    target = TARGET_COL if TARGET_COL in df.columns else "passed_next_stage"
    df = df[df[target].notna()].copy()
    df["age_group"] = df["age"].apply(age_group)
    df["country"] = df["phone"].apply(get_country)
    return df, target

def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    df, target = load_full_data()
    
    # Charger Assets
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEAT_COLS_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    X = df[feature_cols].values.astype(float)
    X_s = scaler.transform(X)
    y_true = df[target].values.astype(int)
    y_proba = model.predict_proba(X_s)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    with open(REPORTS_DIR / "audit.txt", "w", encoding="utf-8") as f:
        f.write("RAPPORT D'AUDIT FINAL — CV-Intelligence (v6)\n")
        f.write("="*60 + "\n\n")

        # 1. Biais Structurels
        f.write("1. BIAIS STRUCTURELS (Sampling)\n" + "="*30 + "\n")
        f.write(f"Nombre total de CV audités : {len(df)}\n")
        f.write(f"Distribution Genre : {df['gender'].value_counts().to_dict()}\n")
        f.write(f"Distribution Age : {df['age_group'].value_counts().to_dict()}\n")
        f.write(f"Distribution Pays : {df['country'].value_counts().to_dict()}\n")
        if "Senior (>45)" not in df["age_group"].values:
            f.write("!! ALERTE : Absence totale de profils Senior (>45) dans le dataset.\n")
        f.write("\n")

        # 2. Analyse d'Équité
        f.write("2. ANALYSE D'ÉQUITÉ (Recall par Groupe)\n" + "="*30 + "\n")
        for col in ["gender", "age_group", "country"]:
            f.write(f"-- Par {col} --\n")
            # Trier par valeur pour la consistance
            unique_vals = sorted(df[col].unique().tolist())
            for val in unique_vals:
                mask = df[col] == val
                yt = y_true[mask]
                yp = y_pred[mask]
                if len(yt) < 1: continue 
                rec = recall_score(yt, yp, zero_division=0)
                prec = precision_score(yt, yp, zero_division=0)
                f.write(f"  {str(val):20} n={len(yt):3}  Recall={rec:.3f}  Precision={prec:.3f}\n")
            f.write("\n")

        # 3. SHAP Explicabilité
        f.write("3. IMPORTANCE DES VARIABLES (SHAP)\n" + "="*30 + "\n")
        explainer = shap.LinearExplainer(model, X_s)
        shap_values = explainer.shap_values(X_s)
        mean_shap = np.abs(shap_values).mean(axis=0)
        feat_imp = pd.Series(mean_shap, index=feature_cols).sort_values(ascending=False)
        for feat, val in feat_imp.items():
            f.write(f"  {feat:25} {val:.4f}\n")

    print(f"Audit final terminé sur le bon dataset (Features + Identities). Rapport : {REPORTS_DIR / 'audit.txt'}")

if __name__ == "__main__":
    main()
