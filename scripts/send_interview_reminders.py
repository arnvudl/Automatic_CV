#!/usr/bin/env python3
"""
send_interview_reminders.py
────────────────────────────
Envoie un email de rappel (via testmail.app) aux candidats
qui ont un entretien aujourd'hui (lu depuis la DB).

Usage :
    python scripts/send_interview_reminders.py
    python scripts/send_interview_reminders.py --dry-run

Cron (tous les jours à 8h) :
    0 8 * * * cd /root/Automatic_CV && python3 scripts/send_interview_reminders.py >> /var/log/reminders.log 2>&1
"""

import os
import sys
import json
import logging
import requests
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("reminders")

TESTMAIL_API_KEY  = os.getenv("TESTMAIL_API_KEY", "")
TESTMAIL_NS       = os.getenv("TESTMAIL_NAMESPACE", "")
FROM_EMAIL        = f"{TESTMAIL_NS}.rh@inbox.testmail.app"
TESTMAIL_SEND_URL = "https://api.testmail.app/api/send"

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/cv_intelligence.db")

sys.path.insert(0, str(Path(__file__).parent.parent))
from sqlalchemy import create_engine, text
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)


def send_reminder(to: str, name: str, interview_type: str, time: str, notes: str) -> bool:
    body = f"""Bonjour {name},

Rappel : vous avez un entretien prévu AUJOURD'HUI.

Type  : {interview_type or 'Entretien'}
Heure : {time or 'À confirmer'}
{f'Notes : {notes}' if notes else ''}

Pour toute question, répondez à cet email.

Bonne chance !
L'équipe RH — Luminary"""

    try:
        r = requests.post(TESTMAIL_SEND_URL, json={
            "apikey":  TESTMAIL_API_KEY,
            "namespace": TESTMAIL_NS,
            "to":      to,
            "from":    FROM_EMAIL,
            "subject": f"Rappel entretien — {date.today().strftime('%d/%m/%Y')}",
            "text":    body,
        }, timeout=10)
        r.raise_for_status()
        log.info(f"  ✓ Envoyé → {to}")
        return True
    except Exception as e:
        log.error(f"  ✗ Échec → {to} : {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not TESTMAIL_API_KEY:
        log.error("TESTMAIL_API_KEY manquant dans .env")
        sys.exit(1)

    today = date.today().strftime("%Y-%m-%d")
    log.info(f"{'[DRY-RUN] ' if args.dry_run else ''}Entretiens du {today}…")

    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT i.candidate_name, i.time, i.interview_type, i.notes,
                       c.email, c.name AS c_name
                FROM interviews i
                LEFT JOIN candidates c ON c.candidate_id = i.candidate_id
                WHERE i.date = :today
            """), {"today": today}).fetchall()
    except Exception as e:
        log.error(f"DB indisponible : {e}")
        sys.exit(1)

    if not rows:
        log.info("Aucun entretien aujourd'hui.")
        return

    sent = errors = skipped = 0
    for r in rows:
        name  = r.candidate_name or r.c_name or "Candidat"
        email = r.email or ""
        if not email:
            log.warning(f"  ⚠ Pas d'email pour {name} — ignoré")
            skipped += 1
            continue

        log.info(f"  → {name} <{email}> — {r.interview_type or 'Entretien'} à {r.time or '?'}")
        if args.dry_run:
            sent += 1
            continue

        if send_reminder(email, name, r.interview_type or "", r.time or "", r.notes or ""):
            sent += 1
        else:
            errors += 1

    log.info(f"Terminé : {sent} envoyé(s), {errors} erreur(s), {skipped} ignoré(s).")


if __name__ == "__main__":
    main()
