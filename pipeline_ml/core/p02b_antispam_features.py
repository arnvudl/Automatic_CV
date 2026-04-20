"""
p02b_antispam_features.py — Features anti-spam
Calcule cv_completeness et red_flag_count depuis les CVs bruts
et les ajoute à features.csv.

cv_completeness : fraction des 8 champs attendus présents (0.0 → 1.0)
red_flag_count  : nb de signaux d'alerte (0 → 4)
  1. Gap emploi > 12 mois entre deux postes consécutifs
  2. years_experience == 0 alors que nb_jobs > 0 (dates manquantes/malformées)
  3. Pas de Target Role (candidature générique/spam)
  4. Aucune compétence déclarée (total_skills == 0)
"""

import re
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent.parent
RAW_DIR       = ROOT / "data" / "raw"
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"

TODAY = datetime.today()

_SEP      = r"(?:\s*[-—]\s*)"
RE_JOB    = re.compile(
    r"(?P<title>.+?)" + _SEP + r"(?P<company>.+?)" + _SEP + r".+?" + _SEP +
    r"(?P<start>[A-Za-z0-9\-]+)\s+to\s+(?P<end>[A-Za-z0-9\-]+)"
)
RE_SECTIONS  = re.compile(r"(Education|Experience|Skills|Languages):", re.IGNORECASE)
RE_NAME      = re.compile(r"^Name:\s*\S+",      re.IGNORECASE | re.MULTILINE)
RE_EMAIL     = re.compile(r"^Email:\s*\S+",     re.IGNORECASE | re.MULTILINE)
RE_PHONE     = re.compile(r"^Phone:\s*\S+",     re.IGNORECASE | re.MULTILINE)
RE_ROLE      = re.compile(r"^Target Role:\s*\S+", re.IGNORECASE | re.MULTILINE)
RE_TECH      = re.compile(r"Technical:\s*(.*)", re.IGNORECASE)
RE_METH      = re.compile(r"Methods:\s*(.*)",   re.IGNORECASE)
RE_MAN       = re.compile(r"Management:\s*(.*)", re.IGNORECASE)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def _parse_date(s: str) -> datetime | None:
    s = s.strip().lower()
    if s == "present":
        return TODAY
    try:
        return datetime.strptime(s, "%Y-%m")
    except ValueError:
        return None

def _sections(content: str) -> dict:
    parts = RE_SECTIONS.split(content)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}

def compute_antispam(filepath: Path) -> dict:
    content = filepath.read_text(encoding="utf-8")
    sections = _sections(content)

    # ── cv_completeness ──────────────────────────────────────────
    checks = [
        bool(RE_NAME.search(content)),
        bool(RE_EMAIL.search(content)),
        bool(RE_PHONE.search(content)),
        bool(RE_ROLE.search(content)),
        "Education"   in sections and bool(sections["Education"].strip()),
        "Experience"  in sections and bool(sections["Experience"].strip()),
        "Skills"      in sections and bool(sections["Skills"].strip()),
        "Languages"   in sections and bool(sections["Languages"].strip()),
    ]
    cv_completeness = round(sum(checks) / len(checks), 3)

    # ── red_flag_count ───────────────────────────────────────────
    flags = 0

    # Flag 3 : pas de Target Role
    if not RE_ROLE.search(content):
        flags += 1

    # Flag 4 : aucune compétence
    total_skills = 0
    if "Skills" in sections:
        sk = sections["Skills"]
        for rx in [RE_TECH, RE_METH, RE_MAN]:
            m = rx.search(sk)
            if m and m.group(1).strip():
                total_skills += len([x for x in m.group(1).split(",") if x.strip()])
    if total_skills == 0:
        flags += 1

    # Extraire tous les jobs avec dates
    job_periods: list[tuple[datetime, datetime]] = []
    if "Experience" in sections:
        for m in RE_JOB.finditer(sections["Experience"]):
            s = _parse_date(m.group("start"))
            e = _parse_date(m.group("end"))
            if s and e and e >= s:
                job_periods.append((s, e))

    nb_jobs = len(job_periods)
    years_exp = sum((e - s).days / 365.25 for s, e in job_periods)

    # Flag 2 : jobs déclarés mais expérience = 0 (dates manquantes)
    exp_section_has_lines = (
        "Experience" in sections
        and len([l for l in sections["Experience"].split("\n") if l.strip()]) > 0
    )
    if exp_section_has_lines and nb_jobs > 0 and years_exp == 0:
        flags += 1

    # Flag 1 : gap > 12 mois entre deux postes consécutifs
    if nb_jobs >= 2:
        sorted_jobs = sorted(job_periods, key=lambda x: x[0])
        for i in range(1, len(sorted_jobs)):
            prev_end   = sorted_jobs[i - 1][1]
            next_start = sorted_jobs[i][0]
            gap_months = (next_start - prev_end).days / 30.44
            if gap_months > 12:
                flags += 1
                break  # un seul flag même si plusieurs gaps

    return {
        "cv_completeness": cv_completeness,
        "red_flag_count":  flags,
    }

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    df = pd.read_csv(FEATURES_PATH)

    results = []
    errors  = 0
    for filepath in sorted(RAW_DIR.glob("*.txt")):
        stem  = filepath.stem
        cv_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, stem))
        try:
            antispam = compute_antispam(filepath)
            results.append({"cv_id": cv_id, **antispam})
        except Exception as exc:
            print(f"  [ERREUR] {filepath.name} : {exc}")
            errors += 1

    df_anti = pd.DataFrame(results)
    print(f"Parsé : {len(df_anti)} CVs  |  Erreurs : {errors}")
    print(f"cv_completeness — moy={df_anti['cv_completeness'].mean():.3f}  "
          f"min={df_anti['cv_completeness'].min():.3f}")
    print(f"red_flag_count  — moy={df_anti['red_flag_count'].mean():.2f}  "
          f"max={df_anti['red_flag_count'].max()}")
    print(df_anti["red_flag_count"].value_counts().sort_index().to_string())

    # Supprimer les anciennes colonnes si déjà présentes (ré-exécution propre)
    for col in ["cv_completeness", "red_flag_count"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df = df.merge(df_anti, on="cv_id", how="left")
    df.to_csv(FEATURES_PATH, index=False)
    print(f"\nfeatures.csv mis à jour ({len(df)} lignes, {len(df.columns)} colonnes).")

if __name__ == "__main__":
    main()
