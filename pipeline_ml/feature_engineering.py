"""
feature_engineering.py — Ingénierie de features pour CV-Intelligence

Enrichit features.csv avec des features composites plus discriminantes.
Toutes les nouvelles features sont dérivées des colonnes existantes,
sans utiliser label, heuristic_score, ni aucune donnée d'identité.

Nouvelles features ajoutées :
  log_years_exp       — log1p(years_experience) : réduit la dominance des outliers
  exp_edu_score       — years_experience * education_level : séniorité qualifiée
  cert_density        — nb_certifications / nb_jobs : certifications par poste
  multilingual_score  — nb_languages + bonus niveau anglais : signal langue composite
  method_tech_ratio   — méthodes / tech : équilibre profil technique vs polyvalent
  tech_per_year       — compétences techniques par année : contrôle le "skill inflation"
  career_depth        — years_experience * avg_job_duration : séniorité + stabilité
  is_it / is_finance  — one-hot secteur (IT et Finance représentent 99% du dataset)
"""

import numpy as np
import pandas as pd
from pathlib import Path

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.csv"

# Colonnes réservées — ne jamais passer au modèle ML
NON_FEATURE_COLS = {
    "cv_id", "profile_type", "target_role", "sector", "education_field",
    "career_progression", "has_english", "has_luxembourgish",
    "label", "heuristic_score",
}

NEW_FEATURES = [
    "log_years_exp",
    "exp_edu_score",
    "cert_density",
    "multilingual_score",
    "method_tech_ratio",
    "tech_per_year",
    "career_depth",
    "is_it",
    "is_finance",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    years    = pd.to_numeric(df["years_experience"],     errors="coerce").fillna(0)
    edu      = pd.to_numeric(df["education_level"],      errors="coerce").fillna(1)
    nb_jobs  = pd.to_numeric(df["nb_jobs"],              errors="coerce").fillna(1).clip(lower=1)
    avg_dur  = pd.to_numeric(df["avg_job_duration"],     errors="coerce").fillna(0)
    nb_cert  = pd.to_numeric(df["nb_certifications"],    errors="coerce").fillna(0)
    nb_lang  = pd.to_numeric(df["nb_languages"],         errors="coerce").fillna(0)
    eng_lvl  = pd.to_numeric(df["english_level"],        errors="coerce").fillna(0)
    nb_tech  = pd.to_numeric(df["nb_technical_skills"],  errors="coerce").fillna(0)
    nb_meth  = pd.to_numeric(df["nb_methods_skills"],    errors="coerce").fillna(0)

    # log1p(experience) : atténue l'avantage des très seniors sans pénaliser les juniors
    df["log_years_exp"] = np.log1p(years).round(3)

    # Expérience × diplôme : un Master avec 8 ans > un Bachelor avec 8 ans
    df["exp_edu_score"] = (years * edu).round(2)

    # Certifications par poste : qualité vs quantité
    df["cert_density"] = (nb_cert / nb_jobs).round(3)

    # Score langues composite : chaque langue compte + bonus anglais courant (B2+)
    eng_bonus = (eng_lvl >= 4).astype(int)
    df["multilingual_score"] = (nb_lang + eng_bonus).round(0)

    # Ratio méthodes/technique : profil équilibré vs pur technicien
    df["method_tech_ratio"] = (nb_meth / nb_tech.clip(lower=1)).round(3)

    # Compétences tech par année : détecte le "CV gonflé" des juniors
    df["tech_per_year"] = (nb_tech / years.clip(lower=0.5)).round(3)

    # Profondeur de carrière : longue + stable = signal senior fiable
    df["career_depth"] = (years * avg_dur).round(2)

    # Secteur one-hot (IT = 80% du dataset, Finance = 21%)
    sector = df["sector"].fillna("Other")
    df["is_it"]      = (sector == "IT").astype(int)
    df["is_finance"]  = (sector == "Finance").astype(int)

    return df


def main():
    if not FEATURES_PATH.exists():
        print(f"Fichier introuvable : {FEATURES_PATH}")
        print("Lance d'abord parse_cv.py")
        return

    df = pd.read_csv(FEATURES_PATH, dtype=str)

    # Supprimer les anciennes features engineered si re-run
    df = df.drop(columns=[c for c in NEW_FEATURES if c in df.columns], errors="ignore")

    df_eng = engineer(df)

    df_eng.to_csv(FEATURES_PATH, index=False)

    print(f"Features ajoutees : {NEW_FEATURES}")
    print(f"Total colonnes    : {len(df_eng.columns)}")
    print(f"Fichier mis a jour : {FEATURES_PATH}")

    # Corrélations avec le label pour validation rapide
    labeled = df_eng[pd.to_numeric(df_eng["label"], errors="coerce").notna()].copy()
    labeled["label"] = pd.to_numeric(labeled["label"])
    print("\nCorrelation avec label (nouvelles features) :")
    for c in NEW_FEATURES:
        r = pd.to_numeric(labeled[c]).corr(labeled["label"])
        marker = " <-- meilleur" if abs(r) > 0.25 else ""
        print(f"  {c:<25} r={r:+.3f}{marker}")


if __name__ == "__main__":
    main()
