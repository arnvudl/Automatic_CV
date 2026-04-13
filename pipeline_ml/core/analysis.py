"""
analysis.py — Analyse avancée des features (VIF, Mutual Information, Heatmap)
"""

import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

# ==============================================================
# CONFIG
# ==============================================================
ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "processed" / "dataset.csv"

FEATURE_COLS = [
    "years_experience", "avg_job_duration", "education_level", "nb_jobs",
    "nb_methods_skills", "nb_languages", "nb_certifications", "english_level",
    "has_german", "nb_technical_skills", "log_years_exp", "exp_edu_score",
    "cert_density", "multilingual_score", "method_tech_ratio", "tech_per_year",
    "career_depth", "is_it", "is_finance"
]

def load_data():
    df = pd.read_csv(DATA_PATH)
    target = "passed_next_stage" if "passed_next_stage" in df.columns else "label"
    df = df[df[target].notna()].copy()
    X = df[FEATURE_COLS].fillna(0)
    y = df[target].astype(int)
    return X, y

def compute_vif(X):
    # Le VIF nécessite un scaler pour être juste
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    return vif_data.sort_values("VIF", ascending=False)

def compute_mi(X, y):
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_data = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores})
    return mi_data.sort_values("MI_Score", ascending=False)

def main():
    X, y = load_data()
    
    print("\n--- ANALYSE VIF (Variance Inflation Factor) ---")
    print("Un VIF > 5-10 indique une forte redondance.")
    vif_df = compute_vif(X)
    print(vif_df)
    
    print("\n--- ANALYSE MUTUAL INFORMATION (MI) ---")
    print("Capture les relations non-linéaires avec la cible.")
    mi_df = compute_mi(X, y)
    print(mi_df)
    
    # Heatmap de Corrélation
    corr = X.corr()
    fig = px.imshow(corr, text_auto=True, title="Heatmap de Corrélation des Features",
                    color_continuous_scale='RdBu_r', aspect="auto")
    
    # Suggestion
    redundant = vif_df[vif_df["VIF"] > 10]["Feature"].tolist()
    useless = mi_df[mi_df["MI_Score"] < 0.01]["Feature"].tolist()
    
    print("\n--- SUGGESTIONS DU CHEF ---")
    if redundant:
        print(f"⚠️ Redondantes (VIF élevé) : {redundant}")
        print("   -> Ces features risquent d'affaiblir la stabilité des coefficients.")
    if useless:
        print(f"📉 Peu informatives (MI bas) : {useless}")
        print("   -> Ces features n'aident probablement pas le modèle.")
    
    # Affichage de la heatmap (optionnel si run via terminal)
    # fig.show()

if __name__ == "__main__":
    main()
