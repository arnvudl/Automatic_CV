"""
p00_exploration.py — Exploration des données brutes (EDA initiale)
À lancer AVANT le feature engineering pour comprendre la qualité des données.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
FEATURES_PATH   = ROOT / "data" / "processed" / "features.csv"
IDENTITIES_PATH = ROOT / "data" / "processed" / "identities.csv"

NUMERIC_COLS = [
    "years_experience", "avg_job_duration", "education_level",
    "nb_jobs", "nb_technical_skills", "nb_methods_skills",
    "nb_management_skills", "total_skills", "nb_languages",
    "english_level", "nb_certifications",
]


def section(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


def check_missing(df: pd.DataFrame) -> None:
    section("1. VALEURS MANQUANTES")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("Aucune valeur manquante.")
    else:
        pct = (missing / len(df) * 100).round(1)
        report = pd.DataFrame({"Manquants": missing, "%": pct})
        print(report.to_string())


def check_class_balance(df: pd.DataFrame, target: str) -> None:
    section("2. ÉQUILIBRE DES CLASSES")
    counts = df[target].value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    report = pd.DataFrame({"Count": counts, "%": pct})
    report.index = report.index.map({0: "Rejeté (0)", 1: "Invité (1)"})
    print(report.to_string())
    ratio = counts.min() / counts.max()
    if ratio < 0.4:
        print(f"\n!! ALERTE déséquilibre : ratio minoritaire/majoritaire = {ratio:.2f}")


def describe_numerics(df: pd.DataFrame) -> None:
    section("3. STATISTIQUES DESCRIPTIVES (numériques)")
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    print(df[cols].describe().round(2).to_string())


def detect_outliers(df: pd.DataFrame) -> None:
    section("4. VALEURS ABERRANTES (méthode IQR)")
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((s < lo) | (s > hi)).sum()
        if n_out > 0:
            rows.append({
                "Feature": col,
                "Outliers": n_out,
                "%": round(n_out / len(s) * 100, 1),
                "Min": round(s.min(), 2),
                "Max": round(s.max(), 2),
                "Limite basse": round(lo, 2),
                "Limite haute": round(hi, 2),
            })
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("Aucun outlier détecté.")


def check_skewness(df: pd.DataFrame) -> None:
    section("5. ASYMÉTRIE (Skewness)")
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    skew = df[cols].apply(pd.to_numeric, errors="coerce").skew().sort_values(ascending=False)
    report = pd.DataFrame({
        "Feature": skew.index,
        "Skewness": skew.values.round(3),
        "Action": ["Log-transform recommandé" if abs(v) > 1 else "OK" for v in skew.values],
    })
    print(report.to_string(index=False))


def check_categoricals(df: pd.DataFrame) -> None:
    section("6. DISTRIBUTIONS CATÉGORIELLES")
    for col in ["sector", "profile_type", "education_field"]:
        if col not in df.columns:
            continue
        print(f"\n--- {col} ---")
        vc = df[col].fillna("(manquant)").value_counts()
        pct = (vc / vc.sum() * 100).round(1)
        print(pd.DataFrame({"Count": vc, "%": pct}).to_string())


def check_identities(df_id: pd.DataFrame) -> None:
    section("7. DONNÉES IDENTITAIRES")
    print(f"CV total : {len(df_id)}")
    for col in ["gender"]:
        if col in df_id.columns:
            vc = df_id[col].fillna("(manquant)").value_counts()
            print(f"\n--- {col} ---")
            print(vc.to_string())
    if "age" in df_id.columns:
        ages = pd.to_numeric(df_id["age"], errors="coerce").dropna()
        print(f"\nÂge — min: {ages.min():.0f}  max: {ages.max():.0f}  "
              f"moyenne: {ages.mean():.1f}  médiane: {ages.median():.1f}")
        missing_age = df_id["age"].isna().sum()
        if missing_age:
            print(f"!! Âge manquant pour {missing_age} CV ({missing_age/len(df_id)*100:.1f}%)")


def main() -> None:
    if not FEATURES_PATH.exists():
        print(f"Fichier introuvable : {FEATURES_PATH}")
        print("Lancez d'abord p01_parse.py.")
        return

    df = pd.read_csv(FEATURES_PATH)
    target = "label" if "label" in df.columns else "passed_next_stage"
    df_labeled = df[df[target].notna()].copy()

    print(f"\nDataset : {len(df)} CV total, {len(df_labeled)} avec label")

    check_missing(df)
    check_class_balance(df_labeled, target)
    describe_numerics(df_labeled)
    detect_outliers(df_labeled)
    check_skewness(df_labeled)
    check_categoricals(df)

    if IDENTITIES_PATH.exists():
        df_id = pd.read_csv(IDENTITIES_PATH)
        check_identities(df_id)
    else:
        print("\n(identities.csv introuvable — section 7 ignorée)")

    print("\nExploration terminée.")


if __name__ == "__main__":
    main()
