"""
audit.py — Audit (Refactored)
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

# ==============================================================
# CONFIG (Adaptée pour core/audit.py)
# ==============================================================
ROOT = Path(__file__).parent.parent.parent
FEATURES_PATH   = ROOT / "data" / "processed" / "dataset.csv"
MODEL_PATH      = ROOT / "models" / "model.pkl"
FEAT_COLS_PATH  = ROOT / "models" / "feature_cols.pkl"
REPORTS_DIR     = ROOT / "reports"

TARGET_COL = "passed_next_stage"

def age_group(age):
    try:
        a = float(age)
        if a < 30: return "Jeune (<30)"
        if a <= 45: return "Adulte (30-45)"
        return "Senior (>45)"
    except: return "Inconnu"

def load():
    global TARGET_COL
    df = pd.read_csv(FEATURES_PATH)
    if TARGET_COL not in df.columns and "label" in df.columns:
        TARGET_COL = "label"
    df["age_group"] = df["age"].apply(age_group)
    return df

def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    df = load()
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEAT_COLS_PATH)

    df_labeled = df[df[TARGET_COL].notna()].copy()
    X = df_labeled[feature_cols].values.astype(float)
    y_true = df_labeled[TARGET_COL].values
    y_pred = model.predict(X)

    print(f"Audit réalisé sur {len(df_labeled)} CV.")
    # (Le reste des calculs de biais peut être ajouté ici ou lu depuis le rapport audit.txt)

if __name__ == "__main__":
    main()
