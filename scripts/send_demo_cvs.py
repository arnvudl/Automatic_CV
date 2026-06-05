#!/usr/bin/env python3
"""
send_demo_cvs.py
─────────────────
Envoie N CVs du dossier data/raw/ en pièce jointe à l'adresse du projet
pour peupler l'ATS avant la présentation.

Usage :
    python scripts/send_demo_cvs.py              # envoie 75 CVs
    python scripts/send_demo_cvs.py --n 30       # envoie 30 CVs
    python scripts/send_demo_cvs.py --dry-run    # aperçu sans envoi
    python scripts/send_demo_cvs.py --delay 10   # pause 10s entre chaque (évite spam)
"""

import os
import sys
import time
import random
import smtplib
import argparse
import logging
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo_sender")

# ── Config ───────────────────────────────────────────────────────────
SMTP_HOST     = os.getenv("SMTP_HOST",     "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER",     "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM     = os.getenv("SMTP_FROM",     SMTP_USER)

# Adresse inbox du projet (n8n écoute dessus)
ATS_INBOX     = "73cn1.rh@inbox.testmail.app"

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"


# ── Envoi ────────────────────────────────────────────────────────────

def send_cv(cv_path: Path, index: int, total: int, dry_run: bool) -> bool:
    """Envoie un CV en pièce jointe à l'inbox ATS."""
    stem = cv_path.stem  # ex: cv_0042

    msg            = MIMEMultipart()
    msg["Subject"] = f"Candidature — {stem}"
    msg["From"]    = SMTP_FROM
    msg["To"]      = ATS_INBOX

    msg.attach(MIMEText(
        f"Bonjour,\n\nVeuillez trouver ci-joint mon CV pour le poste qui m'intéresse.\n\nCordialement",
        "plain", "utf-8"
    ))

    # Pièce jointe
    with cv_path.open("rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{cv_path.name}"')
    msg.attach(part)

    if dry_run:
        log.info(f"  [{index}/{total}] [DRY-RUN] {cv_path.name} → {ATS_INBOX}")
        return True

    try:
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
                smtp.login(SMTP_USER, SMTP_PASSWORD)
                smtp.sendmail(SMTP_FROM, [ATS_INBOX], msg.as_bytes())
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(SMTP_USER, SMTP_PASSWORD)
                smtp.sendmail(SMTP_FROM, [ATS_INBOX], msg.as_bytes())
        log.info(f"  [{index}/{total}] ✓ {cv_path.name} envoyé")
        return True
    except Exception as e:
        log.error(f"  [{index}/{total}] ✗ {cv_path.name} — {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=75,  help="Nombre de CVs à envoyer (défaut: 75)")
    parser.add_argument("--delay",   type=int, default=5,   help="Pause en secondes entre chaque envoi (défaut: 5)")
    parser.add_argument("--dry-run", action="store_true",   help="Aperçu sans envoi")
    parser.add_argument("--seed",    type=int, default=42,  help="Seed pour la sélection aléatoire (reproductible)")
    args = parser.parse_args()

    if not args.dry_run and (not SMTP_USER or not SMTP_PASSWORD):
        log.error("SMTP_USER et SMTP_PASSWORD requis dans .env")
        sys.exit(1)

    # Récupère tous les .txt de data/raw/
    all_cvs = sorted(RAW_DIR.glob("*.txt"))
    if not all_cvs:
        log.error(f"Aucun fichier .txt trouvé dans {RAW_DIR}")
        sys.exit(1)

    # Sélection aléatoire reproductible
    random.seed(args.seed)
    selected = random.sample(all_cvs, min(args.n, len(all_cvs)))
    selected.sort()

    log.info(f"{'[DRY-RUN] ' if args.dry_run else ''}Envoi de {len(selected)} CVs → {ATS_INBOX}")
    log.info(f"Délai entre envois : {args.delay}s — durée estimée : ~{len(selected) * args.delay // 60} min")
    log.info("")

    sent = errors = 0
    for i, cv_path in enumerate(selected, 1):
        ok = send_cv(cv_path, i, len(selected), args.dry_run)
        if ok:
            sent += 1
        else:
            errors += 1

        if not args.dry_run and i < len(selected):
            time.sleep(args.delay)

    log.info("")
    log.info(f"{'Simulé' if args.dry_run else 'Envoyé'} : {sent} | Erreurs : {errors}")
    if not args.dry_run:
        log.info(f"n8n récupère les emails toutes les 2 min → ATS peuplé dans ~{len(selected) * 2 // 60 + 5} min")


if __name__ == "__main__":
    main()
