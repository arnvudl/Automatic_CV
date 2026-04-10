"""
pseudo_labels.py — Génération de pseudo-labels pour TechCore Liège (v2)

Scoring revu pour neutralité par âge :
  - L'expérience est évaluée par sa QUALITE (durée moyenne par poste)
    et non par sa QUANTITE absolue (qui pénalise mécaniquement les jeunes)
  - Les langues sont fortement valorisées (signal RH réel)
  - L'expérience brute reste un bonus mineur, pas le critère dominant
  - Max théorique : ~18 pts. Seuil d'invitation : >= 9

Règle de fairness appliquée : un junior avec bon diplôme, bonne durée
de poste et plusieurs langues doit pouvoir rivaliser avec un senior moyen.
"""

import csv
from pathlib import Path

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
THRESHOLD = 11.5


def score_cv(row: dict) -> float:
    s = 0.0

    # --- Qualité de l'expérience (durée moyenne par poste) ---
    # Récompense la stabilité et l'engagement, indépendant de l'âge.
    avg_dur = float(row.get("avg_job_duration") or 0)
    if avg_dur >= 3.0:
        s += 3    # postes longs, engagement fort
    elif avg_dur >= 1.5:
        s += 2    # durée correcte
    elif avg_dur >= 0.5:
        s += 1    # au moins une vraie expérience

    # --- Bonus séniorité (mineur, non pénalisant pour les jeunes) ---
    years = float(row.get("years_experience") or 0)
    if years >= 7:
        s += 2    # bonus séniorité confirmée
    elif years >= 3:
        s += 1    # bonus expérience intermédiaire
    # Pas de pénalité pour < 3 ans

    # --- Education ---
    edu = int(row.get("education_level") or 1)
    if edu >= 4:
        s += 3    # PhD
    elif edu == 3:
        s += 2    # Master
    elif edu == 2:
        s += 1    # Bachelor

    # --- Langues (signal RH fort, multilinguisme = atout réel) ---
    nb_lang = int(row.get("nb_languages") or 0)
    if nb_lang >= 4:
        s += 3
    elif nb_lang >= 3:
        s += 2
    elif nb_lang >= 2:
        s += 1

    # Niveau d'anglais (CECRL)
    eng_lvl = int(row.get("english_level") or 0)
    if eng_lvl >= 5:    # C1 ou C2
        s += 2
    elif eng_lvl >= 3:  # B1 ou B2
        s += 1

    # --- Compétences techniques ---
    nb_tech = int(row.get("nb_technical_skills") or 0)
    if nb_tech >= 6:
        s += 2
    elif nb_tech >= 3:
        s += 1

    # --- Compétences méthodes ---
    if int(row.get("nb_methods_skills") or 0) >= 3:
        s += 1

    # --- Certifications ---
    nb_cert = int(row.get("nb_certifications") or 0)
    if nb_cert >= 2:
        s += 1
    elif nb_cert >= 1:
        s += 0.5

    # --- Secteur IT (contexte TechCore) ---
    if str(row.get("sector") or "").strip() == "IT":
        s += 1

    return s


def main():
    if not FEATURES_PATH.exists():
        print(f"Fichier introuvable : {FEATURES_PATH}")
        print("Lance d'abord parse_cv.py pour generer features.csv")
        return

    rows = []
    with FEATURES_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    nb_already_labelled = sum(1 for r in rows if r.get("label") not in (None, ""))
    if nb_already_labelled > 0:
        print(f"{nb_already_labelled} CV ont deja un label reel -> conserves tels quels.")

    # Ajouter heuristic_score aux colonnes si absent
    if "heuristic_score" not in fieldnames:
        fieldnames = list(fieldnames) + ["heuristic_score"]

    nb_invited = nb_rejected = nb_skipped = 0
    scores = []

    for row in rows:
        s = score_cv(row)
        row["heuristic_score"] = round(s, 2)
        scores.append(s)

        # Ne pas écraser un vrai label recruteur déjà présent
        if row.get("label") not in (None, ""):
            nb_skipped += 1
            continue

        if s >= THRESHOLD:
            row["label"] = 1
            nb_invited += 1
        else:
            row["label"] = 0
            nb_rejected += 1

    # Réécriture du CSV avec labels + scores
    with FEATURES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_scores = [s for s in scores if s == s]  # exclure nan
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    print(f"CV traites     : {len(rows)}")
    print(f"  Invites (1)  : {nb_invited}  ({nb_invited/len(rows)*100:.1f}%)")
    print(f"  Rejetes (0)  : {nb_rejected}  ({nb_rejected/len(rows)*100:.1f}%)")
    print(f"  Deja labels  : {nb_skipped}")
    print(f"Score moyen    : {avg:.1f} / 16  (seuil = {THRESHOLD})")
    print(f"Fichier mis a jour : {FEATURES_PATH}")


if __name__ == "__main__":
    main()
