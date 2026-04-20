"""
p03_analysis.py — Exploration de Données (EDA) & Analyse Statistique v3
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

ROOT      = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "processed" / "features.csv"

V3_FEATURES = [
    "exp_per_year_of_age", "avg_job_duration", "education_level",
    "potential_score", "junior_potential", "has_multiple_languages",
    "career_depth", "is_it", "field_match",
]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} introuvable. Lancez d'abord p01 et p02.")
    df = pd.read_csv(DATA_PATH)
    target = "label" if "label" in df.columns else "passed_next_stage"
    df = df[df[target].notna()].copy()
    available = [f for f in V3_FEATURES if f in df.columns]
    if len(available) < len(V3_FEATURES):
        missing = set(V3_FEATURES) - set(available)
        print(f"  [WARN] Features manquantes : {missing}")
    X = df[available].fillna(0)
    y = df[target].astype(int)
    return X, y


def detect_outliers(X):
    rows = []
    for col in X.columns:
        Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((X[col] < Q1 - 1.5 * IQR) | (X[col] > Q3 + 1.5 * IQR)).sum()
        if n_out > 0:
            rows.append({"Feature": col, "Outliers": n_out,
                         "Min": X[col].min(), "Max": X[col].max(),
                         "Boundaries": (round(Q1 - 1.5 * IQR, 2), round(Q3 + 1.5 * IQR, 2))})
    return pd.DataFrame(rows)


def check_distributions(X):
    skew = X.skew().sort_values(ascending=False)
    df = pd.DataFrame({"Feature": skew.index, "Skewness": skew.values})
    df["Action"] = df["Skewness"].apply(
        lambda x: "Log-transform suggere" if abs(x) > 1 else "OK")
    return df


def compute_vif(X):
    X_s = StandardScaler().fit_transform(X)
    return pd.DataFrame({
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X_s, i) for i in range(X_s.shape[1])]
    }).sort_values("VIF", ascending=False)


def compute_mi(X, y):
    scores = mutual_info_classif(X, y, random_state=42)
    return pd.DataFrame({"Feature": X.columns, "MI_Score": scores}).sort_values(
        "MI_Score", ascending=False)


def main():
    try:
        X, y = load_data()
    except Exception as e:
        print(f"Erreur : {e}")
        return

    print(f"\n--- EDA v3 ({len(X)} lignes, {len(X.columns)} features) ---")

    print("\n[1. Outliers (IQR)]")
    out = detect_outliers(X)
    print(out.to_string(index=False) if not out.empty else "Aucun outlier detecte.")

    print("\n[2. Distributions (Skewness)]")
    print(check_distributions(X).to_string(index=False))

    print("\n[3. VIF - Colinearite]")
    print(compute_vif(X).to_string(index=False))

    print("\n[4. Mutual Info - Signal avec la cible]")
    print(compute_mi(X, y).to_string(index=False))


if __name__ == "__main__":
    main()
