"""
03_analysis.py — Exploration de Données (EDA) & Analyse Statistique v5
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

# ==============================================================
# CONFIG
# ==============================================================
ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "processed" / "features.csv"

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
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier {DATA_PATH} introuvable. Lancez d'abord le parsing et les features.")
    df = pd.read_csv(DATA_PATH)
    target = "label" if "label" in df.columns else "passed_next_stage"
    df = df[df[target].notna()].copy()
    X = df[V5_FEATURES].fillna(0)
    y = df[target].astype(int)
    return X, y

def detect_outliers(X):
    """Détecte les valeurs aberrantes via la méthode IQR."""
    outliers_report = []
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            n_outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
            if n_outliers > 0:
                outliers_report.append({
                    "Feature": col,
                    "Outliers": n_outliers,
                    "Min": X[col].min(),
                    "Max": X[col].max(),
                    "Boundaries": (round(lower_bound, 2), round(upper_bound, 2))
                })
    return pd.DataFrame(outliers_report)

def check_distributions(X):
    """Vérifie l'asymétrie (skewness) pour suggérer des normalisations."""
    skewness = X.skew().sort_values(ascending=False)
    report = pd.DataFrame({"Feature": skewness.index, "Skewness": skewness.values})
    report["Action Suggérée"] = report["Skewness"].apply(
        lambda x: "Log-transform (Hautement asymétrique)" if x > 1 else "Normal (OK)"
    )
    return report

def compute_vif(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    vif_df = pd.DataFrame()
    vif_df["Feature"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    return vif_df.sort_values("VIF", ascending=False)

def compute_mi(X, y):
    mi_scores = mutual_info_classif(X, y, random_state=42)
    return pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores}).sort_values("MI_Score", ascending=False)

def main():
    try:
        X, y = load_data()
    except Exception as e:
        print(f"Erreur : {e}")
        return

    print(f"\n--- EXPLORATION & ANALYSE STATISTIQUE v5 ({len(X)} lignes) ---")
    
    print("\n[1. Détection des Valeurs Aberrantes (Outliers)]")
    outliers = detect_outliers(X)
    if not outliers.empty:
        print(outliers.to_string(index=False))
    else:
        print("Aucun outlier détecté (méthode IQR).")

    print("\n[2. Analyse des Distributions (Skewness)]")
    dist = check_distributions(X)
    print(dist.to_string(index=False))

    print("\n[3. VIF - Redondance & Colinéarité]")
    vif_df = compute_vif(X)
    print(vif_df.to_string(index=False))
    
    print("\n[4. Mutual Info - Signal avec la cible (Importance non-linéaire)]")
    mi_df = compute_mi(X, y)
    print(mi_df.to_string(index=False))

if __name__ == "__main__":
    main()
