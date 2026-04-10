"""
pseudo_labels.py — Génération de pseudo-labels pour TechCore Liège

Lit data/processed/features.csv (colonne label vide)
Applique le scoring métier défini dans docs/entreprise_fictive.md
Ecrit data/processed/features.csv avec la colonne label remplie

Seuil d'invitation : score >= 8 -> label 1, sinon 0
"""

import csv
from pathlib import Path

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
THRESHOLD = 12.0


def score_cv(row: dict) -> float:
    s = 0.0

    # Expérience
    years = float(row.get("years_experience") or 0)
    if years >= 5:
        s += 3
    elif years >= 3:
        s += 2
    elif years >= 1:
        s += 1

    # Education
    edu = int(row.get("education_level") or 1)
    if edu >= 3:
        s += 2
    elif edu == 2:
        s += 1

    # Anglais
    if int(row.get("has_english") or 0):
        s += 1
    if int(row.get("english_level") or 0) >= 5:  # C1 ou C2
        s += 1

    # Compétences techniques
    nb_tech = int(row.get("nb_technical_skills") or 0)
    if nb_tech >= 5:
        s += 2
    elif nb_tech >= 3:
        s += 1

    # Méthodes
    if int(row.get("nb_methods_skills") or 0) >= 3:
        s += 1

    # Management
    if int(row.get("nb_management_skills") or 0) >= 2:
        s += 1

    # Certifications
    nb_cert = int(row.get("nb_certifications") or 0)
    if nb_cert >= 2:
        s += 1
    elif nb_cert >= 1:
        s += 0.5

    # Progression de carrière
    if int(row.get("career_progression") or 0):
        s += 1

    # Secteur IT
    if str(row.get("sector") or "").strip() == "IT":
        s += 1

    # Multilinguisme
    if int(row.get("nb_languages") or 0) >= 2:
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

    nb_invited = nb_rejected = nb_skipped = 0
    scores = []

    for row in rows:
        if row.get("label") not in (None, ""):
            nb_skipped += 1
            scores.append(float("nan"))
            continue

        s = score_cv(row)
        scores.append(s)
        if s >= THRESHOLD:
            row["label"] = 1
            nb_invited += 1
        else:
            row["label"] = 0
            nb_rejected += 1

    # Réécriture du CSV avec labels
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
