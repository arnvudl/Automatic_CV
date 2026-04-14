"""
features.py — Ingénierie de features (Refactored)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mstats

# ==============================================================
# CONFIG (Adaptée pour core/features.py)
# ==============================================================
ROOT = Path(__file__).parent.parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"

NON_FEATURE_COLS = {
    "cv_id", "profile_type", "target_role", "sector", "education_field",
    "career_progression", "has_english", "has_luxembourgish",
    "label", "passed_next_stage", "heuristic_score",
}

NEW_FEATURES = [
    "log_years_exp", "log_avg_job_duration", "has_multiple_languages",
    "junior_potential", "cert_density", "multilingual_score",
    "method_tech_ratio", "tech_per_year", "career_depth", "is_it", "is_finance",
]

def winsorize(series: pd.Series, limits=(0.05, 0.05)) -> pd.Series:
    """Cap les valeurs aberrantes au percentile 5-95."""
    result = mstats.winsorize(series.fillna(0), limits=limits)
    return pd.Series(result, index=series.index)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # years et avg_dur : pas de winsorisation — variance nécessaire pour le modèle
    years    = pd.to_numeric(df["years_experience"],  errors="coerce").fillna(0)
    avg_dur  = pd.to_numeric(df["avg_job_duration"],  errors="coerce").fillna(0)
    # comptages auxiliaires : winsorisation légère pour les outliers
    nb_jobs  = winsorize(pd.to_numeric(df["nb_jobs"],              errors="coerce")).clip(lower=1)
    nb_cert  = winsorize(pd.to_numeric(df["nb_certifications"],    errors="coerce"))
    nb_lang  = pd.to_numeric(df["nb_languages"],         errors="coerce").fillna(0)
    eng_lvl  = pd.to_numeric(df["english_level"],        errors="coerce").fillna(0)
    nb_tech  = winsorize(pd.to_numeric(df["nb_technical_skills"],  errors="coerce"))
    nb_meth  = winsorize(pd.to_numeric(df["nb_methods_skills"],    errors="coerce"))

    df["log_years_exp"]          = np.log1p(years).round(3)
    df["log_avg_job_duration"]   = np.log1p(avg_dur).round(3)
    df["has_multiple_languages"] = (nb_lang >= 2).astype(int)
    # potential_score : valeurs brutes pour préserver la variance du numérateur
    # (un numérateur trop lissé crée une colinéarité avec years_experience)
    nb_cert_raw = pd.to_numeric(df["nb_certifications"], errors="coerce").fillna(0)
    nb_tech_raw = pd.to_numeric(df["nb_technical_skills"], errors="coerce").fillna(0)
    nb_meth_raw = pd.to_numeric(df["nb_methods_skills"], errors="coerce").fillna(0)
    df["potential_score"]    = ((nb_tech_raw + nb_meth_raw + nb_cert_raw) / (years + 1)).round(2)
    is_junior = (years < 3).astype(int)
    df["junior_potential"] = (is_junior * df["potential_score"]).round(2)
    df["cert_density"]     = (nb_cert / nb_jobs).round(3)
    eng_bonus = (eng_lvl >= 4).astype(int)
    df["multilingual_score"]   = (nb_lang + eng_bonus).round(0)
    df["method_tech_ratio"]    = (nb_meth / nb_tech.clip(lower=1)).round(3)
    df["tech_per_year"]        = (nb_tech / years.clip(lower=0.5)).round(3)
    df["career_depth"]         = (years * avg_dur).round(2)
    sector = df["sector"].fillna("Other")
    df["is_it"]      = (sector == "IT").astype(int)
    df["is_finance"] = (sector == "Finance").astype(int)
    return df

def main():
    if not FEATURES_PATH.exists():
        print(f"Fichier introuvable : {FEATURES_PATH}")
        return
    df = pd.read_csv(FEATURES_PATH, dtype=str)
    df = df.drop(columns=[c for c in NEW_FEATURES if c in df.columns], errors="ignore")
    df_eng = engineer(df)
    df_eng.to_csv(FEATURES_PATH, index=False)
    
    label_col = "passed_next_stage" if "passed_next_stage" in df_eng.columns else "label"
    print(f"Features ajoutées. Corrélation calculée sur {label_col}.")

if __name__ == "__main__":
    main()
