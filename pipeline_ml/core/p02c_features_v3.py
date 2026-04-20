"""
p02c_features_v3.py — Feature Engineering v3
Ajoute deux colonnes à features.csv :
  - exp_per_year_of_age : years_experience / max(age - 22, 1)
      Normalise l'expérience par la durée de carrière possible.
      Réduit le biais structurel lié au genre (pauses carrière).
  - field_match : 1 si le domaine d'études correspond au secteur visé
      Signal de pertinence indépendant de years_experience.
"""

import pandas as pd
from pathlib import Path

ROOT          = Path(__file__).parent.parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
IDENTITIES_PATH = ROOT / "data" / "processed" / "identities.csv"

# Mots-clés pour field_match
IT_KEYWORDS      = {"computer", "software", "data", "information", "it", "computing",
                    "engineering", "technology", "networks", "cybersecurity", "ai"}
FINANCE_KEYWORDS = {"finance", "accounting", "economics", "business", "management",
                    "audit", "banking", "insurance", "financial"}
INDUSTRY_KEYWORDS = {"mechanical", "industrial", "production", "logistics", "supply",
                     "manufacturing", "operations", "civil", "chemical"}


def compute_field_match(education_field: str | None, sector: str | None) -> int:
    if not education_field or not sector:
        return 0
    field = education_field.lower()
    sector = (sector or "").upper()
    if sector == "IT"      and any(k in field for k in IT_KEYWORDS):
        return 1
    if sector == "FINANCE" and any(k in field for k in FINANCE_KEYWORDS):
        return 1
    if sector == "INDUSTRY" and any(k in field for k in INDUSTRY_KEYWORDS):
        return 1
    return 0


def main():
    df = pd.read_csv(FEATURES_PATH)
    df_id = pd.read_csv(IDENTITIES_PATH)[["cv_id", "age"]]
    df_id["age"] = pd.to_numeric(df_id["age"], errors="coerce").fillna(30)

    df = df.merge(df_id, on="cv_id", how="left")

    # ── exp_per_year_of_age ───────────────────────────────────────
    career_years = (df["age"] - 22).clip(lower=1)
    df["exp_per_year_of_age"] = (df["years_experience"] / career_years).round(3)

    # ── field_match ───────────────────────────────────────────────
    df["field_match"] = df.apply(
        lambda r: compute_field_match(r.get("education_field"), r.get("sector")), axis=1
    )

    # Diagnostic rapide
    print("=== exp_per_year_of_age ===")
    print(df["exp_per_year_of_age"].describe().round(3))

    print("\n=== field_match ===")
    print(df["field_match"].value_counts())
    print(f"Taux match : {df['field_match'].mean():.1%}")

    if "label" in df.columns:
        print("\n=== Corrélation avec label ===")
        for feat in ["exp_per_year_of_age", "field_match"]:
            corr = df[feat].corr(df["label"])
            print(f"  {feat:25} corr={corr:+.3f}")

    # Sauvegarder (sans la colonne age temporaire si elle existait déjà)
    # On retire 'age' seulement si elle n'était pas dans le CSV original
    original_cols = list(pd.read_csv(FEATURES_PATH, nrows=0).columns)
    cols_to_drop = [c for c in ["age"] if c not in original_cols]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    df.to_csv(FEATURES_PATH, index=False)
    print(f"\nfeatures.csv mis à jour : {len(df)} lignes, {len(df.columns)} colonnes.")
    print(f"Nouvelles colonnes : exp_per_year_of_age, field_match")


if __name__ == "__main__":
    main()
