"""
audit.py — Audit biais, fairness et explicabilité SHAP pour CV-Intelligence

Analyses réalisées :
  1. Biais structurels (sampling, measurement, omission, representation gap)
  2. Parité démographique et égalité des chances (Fairlearn)
  3. Analyse par sous-groupe : genre, tranche d'âge, localité (indicatif tél.)
     → matrice de confusion + métriques par groupe
  4. SHAP : importance globale et locale (Random Forest)

Note : l'audit tourne sur le dataset COMPLET (205 CV) pour avoir
       suffisamment de données par sous-groupe.
       En production, l'audit devra tourner sur un hold-out séparé.
"""

import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ==============================================================
# CONFIG
# ==============================================================
FEATURES_PATH   = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
IDENTITIES_PATH = Path(__file__).parent.parent / "data" / "processed" / "identities.csv"
MODEL_PATH      = Path(__file__).parent.parent / "models" / "model.pkl"
FEAT_COLS_PATH  = Path(__file__).parent.parent / "models" / "feature_cols.pkl"
REPORTS_DIR     = Path(__file__).parent.parent / "reports"

COUNTRY_CODES = {
    "1":   "USA/Canada",
    "33":  "France",
    "32":  "Belgique",
    "49":  "Allemagne",
    "31":  "Pays-Bas",
    "39":  "Italie",
    "34":  "Espagne",
    "351": "Portugal",
    "353": "Irlande",
    "48":  "Pologne",
    "91":  "Inde",
    "234": "Nigeria",
}

# ==============================================================
# UTILITAIRES
# ==============================================================
def phone_to_country(phone: str) -> str:
    if not isinstance(phone, str):
        return "Inconnu"
    m = re.match(r"\+(\d{1,3})-", phone)
    if not m:
        return "Inconnu"
    return COUNTRY_CODES.get(m.group(1), f"+{m.group(1)}")


def age_group(age) -> str:
    try:
        a = float(age)
        if a < 30:   return "Jeune (<30)"
        if a <= 45:  return "Adulte (30-45)"
        return "Senior (>45)"
    except (TypeError, ValueError):
        return "Inconnu"


def metrics_row(y_true, y_pred, y_proba=None) -> dict:
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {"n": len(y_true), "note": "sous-groupe trop petit ou monoclasse"}
    result = {
        "n":         len(y_true),
        "invite_%":  f"{y_true.mean()*100:.0f}%",
        "predit_%":  f"{y_pred.mean()*100:.0f}%",
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 3),
    }
    if y_proba is not None:
        try:
            result["auc"] = round(roc_auc_score(y_true, y_proba), 3)
        except ValueError:
            result["auc"] = "n/a"
    return result


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def confusion_str(y_true, y_pred) -> str:
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return "  (pas assez de données)"
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[-1,-1])
    return (
        f"  Vrais Negatifs (rejete->rejete) : {tn}\n"
        f"  Faux Positifs  (rejete->invite) : {fp}\n"
        f"  Faux Negatifs  (invite->rejete) : {fn}\n"
        f"  Vrais Positifs (invite->invite) : {tp}"
    )


# ==============================================================
# 1. CHARGEMENT
# ==============================================================
def load() -> pd.DataFrame:
    feat = pd.read_csv(FEATURES_PATH)
    iden = pd.read_csv(IDENTITIES_PATH)
    df = feat.merge(iden[["cv_id", "gender", "age", "phone"]], on="cv_id", how="left")
    df["age_group"] = df["age"].apply(age_group)
    df["country"]   = df["phone"].apply(phone_to_country)
    return df


# ==============================================================
# 2. BIAIS STRUCTURELS
# ==============================================================
def audit_structural_bias(df: pd.DataFrame, lines: list):
    print_section("1. BIAIS STRUCTURELS")
    lines.append("=" * 60)
    lines.append("1. BIAIS STRUCTURELS")
    lines.append("=" * 60)

    n = len(df)

    # --- Sampling bias ---
    gender_counts = df["gender"].value_counts()
    gender_ratio  = gender_counts.min() / gender_counts.max()
    flag_gender   = "!!" if gender_ratio < 0.8 else "OK"

    age_counts  = df["age_group"].value_counts()
    has_senior  = "Senior (>45)" in age_counts.index and age_counts.get("Senior (>45)", 0) > 0
    flag_senior = "!! ABSENT" if not has_senior else "OK"

    country_counts = df["country"].value_counts()
    top_country_share = country_counts.iloc[0] / n

    print("\n[Sampling Bias] Distribution des groupes dans le dataset")
    print(f"  Genre     : {gender_counts.to_dict()}  (ratio minorité/majorité = {gender_ratio:.2f}) {flag_gender}")
    print(f"  Âge       : {age_counts.to_dict()}")
    print(f"  Seniors   : {flag_senior}")
    print(f"  Pays top 5: {country_counts.head(5).to_dict()}")
    print(f"  Part du pays dominant : {top_country_share:.0%}")

    lines += [
        "\n[Sampling Bias]",
        f"  Genre     : {gender_counts.to_dict()}  (ratio = {gender_ratio:.2f}) {flag_gender}",
        f"  Tranches d'age : {age_counts.to_dict()}",
        f"  Seniors   : {flag_senior}",
        f"  Pays top 5: {country_counts.head(5).to_dict()}",
    ]

    # --- Measurement bias ---
    print("\n[Measurement Bias] Qualité des features extraites")
    missing = df[[c for c in df.columns if c not in ("cv_id","source_filename","gender","age","phone","age_group","country","target_role","sector","education_field","label")]].isnull().sum()
    if missing.any():
        print(f"  Valeurs manquantes : {missing[missing>0].to_dict()}")
        lines.append(f"  [Measurement Bias] Manquants : {missing[missing>0].to_dict()}")
    else:
        print("  Aucune valeur manquante dans les features numériques. OK")
        lines.append("  [Measurement Bias] Aucun manquant. OK")

    # Vérifie si years_experience pourrait être sous-estimée
    zero_exp = (df["years_experience"] == 0).sum()
    print(f"  CV avec years_experience = 0 : {zero_exp} ({zero_exp/n:.0%})")
    lines.append(f"  CV avec years_experience = 0 : {zero_exp} ({zero_exp/n:.0%})")

    # --- Omission bias ---
    print("\n[Omission Bias] Features exclues du modèle")
    excluded = ["gender", "age", "name", "email", "phone", "country"]
    print(f"  Exclues intentionnellement (RGPD) : {excluded}")
    print("  education_field : présente mais non utilisée (trop fragmentée)")
    print("  target_role     : présente mais non utilisée (encodage non implémenté)")
    lines += [
        "\n[Omission Bias]",
        f"  Exclues intentionnellement (RGPD) : {excluded}",
        "  education_field et target_role présentes mais non utilisées comme features ML.",
        "  Risque : si ces champs corrèlent avec le label, le modèle rate une information.",
    ]

    # --- Representation gap ---
    print("\n[Representation Gap] Groupes sous-représentés")
    for col, threshold in [("gender", 40), ("age_group", 15), ("country", 5)]:
        small = df[col].value_counts()[df[col].value_counts() < threshold]
        if not small.empty:
            print(f"  {col}: groupes < {threshold} exemples → {small.to_dict()}")
            lines.append(f"  {col}: groupes < {threshold} → {small.to_dict()}")
    if not has_senior:
        print("  CRITIQUE : aucun profil senior (>45 ans) dans le dataset.")
        lines.append("  CRITIQUE : aucun profil senior (>45 ans). Generalisation impossible sur ce groupe.")


# ==============================================================
# 3. FAIRNESS — PARITÉ DÉMOGRAPHIQUE & ÉGALITÉ DES CHANCES
# ==============================================================
def audit_fairness(df: pd.DataFrame, y_pred: np.ndarray, y_proba: np.ndarray, lines: list):
    print_section("2. PARITÉ DÉMOGRAPHIQUE & ÉGALITÉ DES CHANCES")
    lines.append("\n" + "=" * 60)
    lines.append("2. PARITÉ DÉMOGRAPHIQUE & ÉGALITÉ DES CHANCES")
    lines.append("=" * 60)

    y_true = df["label"].values

    for attr in ["gender", "age_group"]:
        print(f"\n-- Par {attr} --")
        lines.append(f"\n-- Par {attr} --")

        rates = {}
        for group in sorted(df[attr].dropna().unique()):
            mask = df[attr] == group
            n_g  = mask.sum()
            if n_g < 5:
                continue
            accept_rate = y_pred[mask].mean()
            true_rate   = y_true[mask].mean()
            rates[group] = accept_rate
            print(f"  {group:<20} n={n_g:3d}  vrais invités={true_rate:.0%}  prédits invités={accept_rate:.0%}")
            lines.append(f"  {group:<20} n={n_g}  vrais={true_rate:.0%}  prédits={accept_rate:.0%}")

        # Disparate Impact Ratio
        if len(rates) >= 2:
            sorted_rates = sorted(rates.values())
            di = sorted_rates[0] / sorted_rates[-1] if sorted_rates[-1] > 0 else 0
            flag = "OK" if di >= 0.80 else "!! EN DESSOUS DU SEUIL 80%"
            print(f"\n  Disparate Impact Ratio : {di:.3f}  {flag}")
            print(f"  (règle des 80% : DI >= 0.80 est légalement acceptable)")
            lines.append(f"  Disparate Impact Ratio : {di:.3f}  {flag}")

        # Equal Opportunity : recall par groupe (taux de vrais positifs)
        print(f"\n  Equal Opportunity (recall par groupe) :")
        lines.append("  Equal Opportunity :")
        recalls = {}
        for group in sorted(df[attr].dropna().unique()):
            mask = df[attr] == group
            if mask.sum() < 5 or y_true[mask].sum() == 0:
                continue
            r = recall_score(y_true[mask], y_pred[mask], zero_division=0)
            recalls[group] = r
            print(f"  {group:<20} recall={r:.3f}")
            lines.append(f"    {group:<20} recall={r:.3f}")
        if len(recalls) >= 2:
            eo_diff = max(recalls.values()) - min(recalls.values())
            flag = "OK" if eo_diff < 0.10 else "!! Ecart > 10%"
            print(f"  Ecart max de recall : {eo_diff:.3f}  {flag}")
            lines.append(f"  Ecart max recall : {eo_diff:.3f}  {flag}")

    # Fairlearn (si disponible)
    try:
        from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
        dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=df["gender"])
        eod = equalized_odds_difference(y_true, y_pred, sensitive_features=df["gender"])
        print(f"\n  [Fairlearn] Demographic Parity Difference (genre) : {dpd:.4f}  (objectif : proche de 0)")
        print(f"  [Fairlearn] Equalized Odds Difference (genre)      : {eod:.4f}  (objectif : proche de 0)")
        lines.append(f"\n  [Fairlearn] DPD genre : {dpd:.4f}   EOD genre : {eod:.4f}")
    except ImportError:
        print("  Fairlearn non disponible.")


# ==============================================================
# 4. ANALYSE PAR SOUS-GROUPE
# ==============================================================
def audit_subgroups(df: pd.DataFrame, y_pred: np.ndarray, y_proba: np.ndarray, lines: list):
    print_section("3. ANALYSE PAR SOUS-GROUPE")
    lines.append("\n" + "=" * 60)
    lines.append("3. ANALYSE PAR SOUS-GROUPE")
    lines.append("=" * 60)

    y_true = df["label"].values

    for attr in ["gender", "age_group", "country"]:
        print(f"\n{'-'*50}")
        print(f"  Attribut : {attr}")
        print(f"{'-'*50}")
        lines.append(f"\n--- {attr} ---")

        for group in sorted(df[attr].dropna().unique()):
            mask   = (df[attr] == group).values
            n_g    = mask.sum()
            if n_g < 5:
                print(f"  {group:<25} n={n_g} — trop petit, ignoré")
                continue

            yt = y_true[mask]
            yp = y_pred[mask]
            ya = y_proba[mask]

            row = metrics_row(yt, yp, ya)
            print(f"\n  Groupe : {group}  (n={n_g})")
            for k, v in row.items():
                print(f"    {k:<12} : {v}")
            print("  Matrice de confusion :")
            print(confusion_str(yt, yp))

            lines.append(f"\n  Groupe : {group}  (n={n_g})")
            for k, v in row.items():
                lines.append(f"    {k} : {v}")
            lines.append("  Matrice de confusion :")
            lines.append(confusion_str(yt, yp))


# ==============================================================
# 5. SHAP
# ==============================================================
def audit_shap(model, X: np.ndarray, feature_cols: list, df: pd.DataFrame, lines: list):
    print_section("4. EXPLICABILITÉ SHAP")
    lines.append("\n" + "=" * 60)
    lines.append("4. EXPLICABILITÉ SHAP")
    lines.append("=" * 60)

    # shap.Explainer auto-detecte le type de modele (tree, linear, kernel...)
    explainer   = shap.Explainer(model, shap.maskers.Independent(X, max_samples=100))
    shap_result = explainer(X)

    # .values peut etre (n, features) ou (n, features, classes)
    sv = shap_result.values
    if sv.ndim == 3:
        sv = sv[:, :, 1]   # classe "invite"

    # --- Importance globale ---
    mean_abs = np.abs(sv).mean(axis=0)
    importance = sorted(zip(feature_cols, mean_abs), key=lambda x: x[1], reverse=True)

    print("\n[Importance globale] Contribution moyenne |SHAP| par feature :")
    lines.append("\n[Importance globale] |SHAP| moyen :")
    for feat, val in importance:
        bar = "#" * int(val * 60)
        print(f"  {feat:<30} {val:.4f}  {bar}")
        lines.append(f"  {feat:<30} {val:.4f}")

    # --- Direction de l'effet (positive ou négative) ---
    mean_signed = sv.mean(axis=0)
    print("\n[Direction des effets] Contribution SHAP signée (moyenne) :")
    lines.append("\n[Direction des effets] SHAP signé moyen :")
    for feat, val in sorted(zip(feature_cols, mean_signed), key=lambda x: x[1], reverse=True):
        direction = "+ favorise invitation" if val > 0 else "- defavorise invitation"
        print(f"  {feat:<30} {val:+.4f}  {direction}")
        lines.append(f"  {feat:<30} {val:+.4f}  {direction}")

    # --- Top 3 profils locaux ---
    print("\n[Importance locale] Explication des 3 meilleurs scores prédits :")
    lines.append("\n[Importance locale] Top 3 scores :")
    proba = model.predict_proba(X)[:, 1]
    top3  = np.argsort(proba)[::-1][:3]

    for rank, idx in enumerate(top3, 1):
        print(f"\n  Candidat #{rank} (cv_id={df.iloc[idx]['cv_id'][:8]}…  score={proba[idx]:.3f})")
        lines.append(f"\n  Candidat #{rank}  score={proba[idx]:.3f}")
        local_shap = sorted(zip(feature_cols, sv[idx]), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feat, val in local_shap:
            raw_val = X[idx, feature_cols.index(feat)]
            direction = "+" if val > 0 else "-"
            print(f"    {direction} {feat:<28} valeur={raw_val}  SHAP={val:+.4f}")
            lines.append(f"    {direction} {feat:<28} val={raw_val}  SHAP={val:+.4f}")

    # --- Bottom 3 ---
    print("\n[Importance locale] Explication des 3 scores les plus bas :")
    lines.append("\n[Importance locale] Bottom 3 scores :")
    bot3 = np.argsort(proba)[:3]

    for rank, idx in enumerate(bot3, 1):
        print(f"\n  Candidat #{rank} (cv_id={df.iloc[idx]['cv_id'][:8]}…  score={proba[idx]:.3f})")
        lines.append(f"\n  Candidat #{rank}  score={proba[idx]:.3f}")
        local_shap = sorted(zip(feature_cols, sv[idx]), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feat, val in local_shap:
            raw_val = X[idx, feature_cols.index(feat)]
            direction = "+" if val > 0 else "-"
            print(f"    {direction} {feat:<28} valeur={raw_val}  SHAP={val:+.4f}")
            lines.append(f"    {direction} {feat:<28} val={raw_val}  SHAP={val:+.4f}")


# ==============================================================
# MAIN
# ==============================================================
def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    # Chargement
    df           = load()
    model        = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEAT_COLS_PATH)

    # Supprimer les CV sans label pour les métriques de fairness
    df_labeled = df[df["label"].notna()].copy()
    unlabeled  = len(df) - len(df_labeled)
    if unlabeled:
        print(f"  {unlabeled} CV sans label exclus des metriques (gardes pour SHAP)")

    X       = df[feature_cols].values.astype(float)
    X_lab   = df_labeled[feature_cols].values.astype(float)
    y_pred  = model.predict(X_lab)
    y_proba = model.predict_proba(X_lab)[:, 1]

    lines = ["RAPPORT D'AUDIT — CV-Intelligence", "=" * 60, ""]

    print(f"\nDataset : {len(df)} CV  ({len(df_labeled)} labelles)")
    print(f"Predit invites : {y_pred.sum()} ({y_pred.mean():.0%})")

    audit_structural_bias(df_labeled, lines)
    audit_fairness(df_labeled, y_pred, y_proba, lines)
    audit_subgroups(df_labeled, y_pred, y_proba, lines)
    audit_shap(model, X, feature_cols, df, lines)  # SHAP sur tous les CV

    # Sauvegarde du rapport
    report_path = REPORTS_DIR / "audit.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n\nRapport sauvegarde : {report_path}")


if __name__ == "__main__":
    main()
