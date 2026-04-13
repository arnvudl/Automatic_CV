"""
audit.py — Audit de Biais & Explicabilité (V4-FAIR)
"""

import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# ==============================================================
# CONFIG
# ==============================================================
ROOT = Path(__file__).parent.parent.parent
FEATURES_PATH   = ROOT / "data" / "processed" / "dataset.csv"
MODEL_PATH      = ROOT / "models" / "model.pkl"
SCALER_PATH     = ROOT / "models" / "scaler.pkl"
FEAT_COLS_PATH  = ROOT / "models" / "feature_cols.pkl"
THRESHOLD_PATH  = ROOT / "models" / "threshold.pkl"
REPORTS_DIR     = ROOT / "reports"

TARGET_COL = "passed_next_stage"

def age_group(age):
    try:
        a = float(age)
        if a < 30: return "Jeune (<30)"
        if a <= 45: return "Adulte (30-45)"
        return "Senior (>45)"
    except: return "Inconnu"

def load_data():
    df = pd.read_csv(FEATURES_PATH)
    target = TARGET_COL if TARGET_COL in df.columns else "label"
    df = df[df[target].notna()].copy()
    df["age_group"] = df["age"].apply(age_group)
    return df, target

def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    df, target = load_data()
    
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
        f.write("RAPPORT D'AUDIT — CV-Intelligence (V4-FAIR)\n")
        f.write("="*60 + "\n\n")

        # 1. Biais Structurels (Sampling)
        f.write("1. BIAIS STRUCTURELS\n" + "="*30 + "\n")
        f.write(f"Genre : {df['gender'].value_counts().to_dict()}\n")
        f.write(f"Age Group : {df['age_group'].value_counts().to_dict()}\n")
        if "Senior (>45)" not in df["age_group"].values:
            f.write("!! ALERTE : Aucun profil Senior (>45) détecté.\n")
        f.write("\n")

        # 2. Analyse par Groupe (Gender & Age)
        for col in ["gender", "age_group"]:
            f.write(f"-- Par {col} --\n")
            for val in df[col].unique():
                mask = df[col] == val
                yt = y_true[mask]
                yp = y_pred[mask]
                if len(yt) == 0: continue
                rec = recall_score(yt, yp, zero_division=0)
                prec = precision_score(yt, yp, zero_division=0)
                f.write(f"  {val:20} n={len(yt):3}  vrais_invités={yt.mean():.1%}  prédits_invités={yp.mean():.1%}\n")
                f.write(f"  {'':20} precision={prec:.3f} recall={rec:.3f}\n")
            f.write("\n")

        # 3. Explicabilité SHAP
        f.write("2. EXPLICABILITÉ SHAP\n" + "="*30 + "\n")
        explainer = shap.LinearExplainer(model, X_s)
        shap_values = explainer.shap_values(X_s)
        
        # Importance Globale
        mean_shap = np.abs(shap_values).mean(axis=0)
        feat_imp = pd.Series(mean_shap, index=feature_cols).sort_values(ascending=False)
        f.write("[Importance Globale |SHAP|]\n")
        for feat, val in feat_imp.items():
            f.write(f"  {feat:25} {val:.4f}\n")
        f.write("\n")

    print(f"Audit terminé. Rapport écrit dans {REPORTS_DIR / 'audit.txt'}")

if __name__ == "__main__":
    main()
