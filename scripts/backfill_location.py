#!/usr/bin/env python3
"""
backfill_location.py
─────────────────────
Relit les fichiers texte bruts des candidats existants pour extraire
leur lieu de résidence (city, country) et met à jour la DB.

Usage :
    python scripts/backfill_location.py
    python scripts/backfill_location.py --dry-run
"""

import re
import sys
import logging
import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("backfill_location")

RE_ADDRESS = re.compile(
    r"(?:Address|Location|Adresse)\s*[:\-]?\s*(.+)",
    re.IGNORECASE,
)

def extract_location(text: str) -> str | None:
    """Extrait 'City, Country' depuis le texte brut d'un CV."""
    m = RE_ADDRESS.search(text)
    if not m:
        return None
    address = m.group(1).strip()
    parts = [p.strip() for p in address.split(",")]
    if len(parts) >= 3:
        # "123 Street, 12345 City, Country"
        city_part = parts[-2].strip()
        city = re.sub(r"^\d+\s*", "", city_part).strip() or city_part
        country = parts[-1].strip()
        return f"{city}, {country}" if city and country else country or city
    elif len(parts) == 2:
        return f"{parts[0].strip()}, {parts[1].strip()}"
    return parts[0].strip() or None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Affiche sans modifier")
    args = parser.parse_args()

    from api.config import RAW_TEXTS_DIR
    from api.database import get_db, Candidate as CandidateModel

    updated = skipped = missing = 0

    with get_db() as db:
        candidates = db.query(CandidateModel).all()
        log.info(f"{len(candidates)} candidats en DB")

        for c in candidates:
            # Déjà rempli → skip
            if c.location:
                skipped += 1
                continue

            raw = RAW_TEXTS_DIR / f"{c.candidate_id}.txt"
            if not raw.exists():
                log.warning(f"  ✗ Pas de fichier brut pour {c.name} ({c.candidate_id[:8]}…)")
                missing += 1
                continue

            text = raw.read_text(encoding="utf-8", errors="replace")
            location = extract_location(text)

            if location:
                log.info(f"  {'[DRY] ' if args.dry_run else ''}→ {c.name} : {location}")
                if not args.dry_run:
                    c.location = location
                updated += 1
            else:
                log.info(f"  ⚠ Adresse non trouvée : {c.name}")
                missing += 1

    log.info(f"\nTerminé — mis à jour : {updated} | déjà remplis : {skipped} | non trouvés : {missing}")


if __name__ == "__main__":
    main()
