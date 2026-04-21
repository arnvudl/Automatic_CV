"""
p02_features.py — Feature Engineering v4
Features de base + features anti-biais :
  - exp_per_year_of_age : years_experience / max(age-22, 1)
  - field_match         : adéquation formation / secteur
  - education_adj       : éducation compressée (Bachelor≈0.3, Master≈0.7)
                          L'analyse des labels montre que Bachelors invités ont
                          career_depth ~41 == Masters invités (~40) → le diplôme
                          ne doit plus dominer, c'est la profondeur de carrière
                          qui compte (RRK-inspired, Google hiring philosophy).
  - potential_per_year  : skills acquis par an (GCA proxy — récompense
                          l'apprentissage rapide chez les juniors)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mstats

ROOT            = Path(__file__).parent.parent.parent
FEATURES_PATH   = ROOT / "data" / "processed" / "features.csv"
IDENTITIES_PATH = ROOT / "data" / "processed" / "identities.csv"

IT_KEYWORDS       = {"computer", "software", "data", "information", "it", "computing",
                     "engineering", "technology", "networks", "cybersecurity", "ai"}
FINANCE_KEYWORDS  = {"finance", "accounting", "economics", "business", "management",
                     "audit", "banking", "insurance", "financial"}
INDUSTRY_KEYWORDS = {"mechanical", "industrial", "production", "logistics", "supply",
                     "manufacturing", "operations", "civil", "chemical"}

NEW_FEATURES = [
    "log_years_exp", "log_avg_job_duration", "has_multiple_languages",
    "potential_score", "junior_potential", "cert_density", "multilingual_score",
    "method_tech_ratio", "tech_per_year", "career_depth", "is_it", "is_finance",
    "exp_per_year_of_age", "field_match",
    "education_adj", "potential_per_year",
]

OBSOLETE_COLS = ["cv_completeness", "red_flag_count"]


def winsorize(series: pd.Series, limits=(0.05, 0.05)) -> pd.Series:
    result = mstats.winsorize(series.fillna(0), limits=limits)
    return pd.Series(result, index=series.index)


def _field_match(education_field, sector) -> int:
    if not education_field or not sector:
        return 0
    field  = str(education_field).lower()
    sector = str(sector).upper()
    if sector == "IT"       and any(k in field for k in IT_KEYWORDS):
        return 1
    if sector == "FINANCE"  and any(k in field for k in FINANCE_KEYWORDS):
        return 1
    if sector == "INDUSTRY" and any(k in field for k in INDUSTRY_KEYWORDS):
        return 1
    return 0


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    years   = pd.to_numeric(df["years_experience"], errors="coerce").fillna(0)
    avg_dur = pd.to_numeric(df["avg_job_duration"],  errors="coerce").fillna(0)
    nb_jobs = winsorize(pd.to_numeric(df["nb_jobs"],             errors="coerce")).clip(lower=1)
    nb_cert = winsorize(pd.to_numeric(df["nb_certifications"],   errors="coerce"))
    nb_lang = pd.to_numeric(df["nb_languages"],      errors="coerce").fillna(0)
    eng_lvl = pd.to_numeric(df["english_level"],     errors="coerce").fillna(0)
    nb_tech = winsorize(pd.to_numeric(df["nb_technical_skills"], errors="coerce"))
    nb_meth = winsorize(pd.to_numeric(df["nb_methods_skills"],   errors="coerce"))

    nb_cert_raw = pd.to_numeric(df["nb_certifications"],   errors="coerce").fillna(0)
    nb_tech_raw = pd.to_numeric(df["nb_technical_skills"], errors="coerce").fillna(0)
    nb_meth_raw = pd.to_numeric(df["nb_methods_skills"],   errors="coerce").fillna(0)

    df["log_years_exp"]        = np.log1p(years).round(3)
    df["log_avg_job_duration"] = np.log1p(avg_dur).round(3)
    df["has_multiple_languages"] = (nb_lang >= 2).astype(int)
    df["potential_score"]      = ((nb_tech_raw + nb_meth_raw + nb_cert_raw) / (years + 1)).round(2)
    is_junior                  = (years < 3).astype(int)
    df["junior_potential"]     = (is_junior * df["potential_score"]).round(2)
    df["cert_density"]         = (nb_cert / nb_jobs).round(3)
    df["multilingual_score"]   = (nb_lang + (eng_lvl >= 4).astype(int)).round(0)
    df["method_tech_ratio"]    = (nb_meth / nb_tech.clip(lower=1)).round(3)
    df["tech_per_year"]        = (nb_tech / years.clip(lower=0.5)).round(3)
    df["career_depth"]         = (years * avg_dur).round(2)
    sector = df["sector"].fillna("Other")
    df["is_it"]      = (sector == "IT").astype(int)
    df["is_finance"] = (sector == "Finance").astype(int)

    # ── Features anti-biais v3 ────────────────────────────────
    if "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce").fillna(30)
    else:
        age = pd.Series(30, index=df.index)
    career_years = (age - 22).clip(lower=1)
    df["exp_per_year_of_age"] = (years / career_years).round(3)

    df["field_match"] = df.apply(
        lambda r: _field_match(r.get("education_field"), r.get("sector")), axis=1
    )

    # ── Features anti-biais v4 (RRK/GCA inspired) ────────────────────────────
    #
    # education_adj : compresse l'échelle 1-4 → 0.0/0.3/0.7/0.8
    # Analyse des labels : Bachelor invité (career_depth=41) ≈ Master invité (40)
    # → le diplôme est un bonus, pas un filtre. Réduire son emprise dans le modèle
    # permet à career_depth et potential d'exprimer leur vraie valeur prédictive.
    edu_raw = pd.to_numeric(df["education_level"], errors="coerce").fillna(2)
    edu_map = {1: 0.0, 2: 0.30, 3: 0.70, 4: 0.80}
    df["education_adj"] = edu_raw.round().astype(int).map(edu_map).fillna(0.30).round(2)

    # potential_per_year : (skills + méthodes + certifs) / années travaillées
    # Proxy GCA (General Cognitive Ability) — récompense la vitesse d'apprentissage.
    # Distinct de potential_score (÷ years+1) : ici on divise par années réelles
    # pour valoriser les juniors qui accumulent des compétences vite.
    df["potential_per_year"] = (
        (nb_tech_raw + nb_meth_raw + nb_cert_raw) / years.clip(lower=0.5)
    ).round(2)

    return df


def main():
    if not FEATURES_PATH.exists():
        print(f"Fichier introuvable : {FEATURES_PATH}")
        return

    df = pd.read_csv(FEATURES_PATH, dtype=str)

    # Fusionner l'âge depuis identities pour exp_per_year_of_age
    if IDENTITIES_PATH.exists():
        df_id = pd.read_csv(IDENTITIES_PATH)[["cv_id", "age"]]
        if "age" in df.columns:
            df.drop(columns=["age"], inplace=True)
        df = df.merge(df_id, on="cv_id", how="left")

    # Supprimer colonnes obsolètes et anciennes features pour recalcul propre
    drop_cols = [c for c in NEW_FEATURES + OBSOLETE_COLS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    df_eng = engineer(df)

    # Retirer la colonne age temporaire (elle est dans identities, pas features)
    df_eng.drop(columns=["age"], inplace=True, errors="ignore")

    df_eng.to_csv(FEATURES_PATH, index=False)
    print(f"Features v3 calculées sur {len(df_eng)} CV ({len(df_eng.columns)} colonnes).")
    print(f"  exp_per_year_of_age : moy={df_eng['exp_per_year_of_age'].astype(float).mean():.3f}")
    print(f"  field_match         : {df_eng['field_match'].astype(int).mean():.1%} de correspondance")


if __name__ == "__main__":
    main()
