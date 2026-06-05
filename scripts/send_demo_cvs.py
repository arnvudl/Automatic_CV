#!/usr/bin/env python3
"""
send_demo_cvs.py
─────────────────
Envoie un batch de CVs par jour vers l'inbox ATS (via testmail.app).
Garde un historique pour ne jamais renvoyer le même CV.

Usage :
    python scripts/send_demo_cvs.py           # envoie le batch du jour (défaut: 15)
    python scripts/send_demo_cvs.py --n 20    # envoie 20 CVs aujourd'hui
    python scripts/send_demo_cvs.py --dry-run # aperçu sans envoi
    python scripts/send_demo_cvs.py --status  # voir combien ont été envoyés

Exemple pour arriver à 75 en 5 jours :
    Jour 1 (6 juin)  : python scripts/send_demo_cvs.py --n 15
    Jour 2 (7 juin)  : python scripts/send_demo_cvs.py --n 15
    Jour 3 (8 juin)  : python scripts/send_demo_cvs.py --n 15
    Jour 4 (9 juin)  : python scripts/send_demo_cvs.py --n 15
    Jour 5 (10 juin) : python scripts/send_demo_cvs.py --n 15
    → 75 candidats dans l'ATS, aucun doublon
"""

import os
import sys
import json
import time
import base64
import random
import logging
import requests
import argparse
from pathlib import Path
from datetime import date

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
TESTMAIL_API_KEY  = os.getenv("TESTMAIL_API_KEY", "")
TESTMAIL_NS       = os.getenv("TESTMAIL_NAMESPACE", "")
ATS_INBOX         = f"{TESTMAIL_NS}.rh@inbox.testmail.app"
FROM_EMAIL        = f"{TESTMAIL_NS}.demo@inbox.testmail.app"
TESTMAIL_SEND_URL = "https://api.testmail.app/api/send"

RAW_DIR      = Path(__file__).parent.parent / "data" / "raw"
STATE_FILE   = Path(__file__).parent.parent / "data" / "demo_sent.json"


# ── Tracking ─────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"sent": [], "total": 0}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Envoi ────────────────────────────────────────────────────────────

def send_cv(cv_path: Path) -> bool:
    content_b64 = base64.b64encode(cv_path.read_bytes()).decode()
    try:
        r = requests.post(TESTMAIL_SEND_URL, json={
            "apikey":    TESTMAIL_API_KEY,
            "namespace": TESTMAIL_NS,
            "to":        ATS_INBOX,
            "from":      FROM_EMAIL,
            "subject":   f"Candidature — {cv_path.stem}",
            "text":      "Bonjour,\n\nVeuillez trouver mon CV en pièce jointe.\n\nCordialement",
            "attachments": [{
                "filename":    cv_path.name,
                "content":     content_b64,
                "contentType": "text/plain",
            }],
        }, timeout=15)
        r.raise_for_status()
        return True
    except Exception as e:
        log.error(f"  ✗ {cv_path.name} : {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=15,  help="CVs à envoyer aujourd'hui (défaut: 15)")
    parser.add_argument("--dry-run", action="store_true",   help="Aperçu sans envoi")
    parser.add_argument("--status",  action="store_true",   help="Voir l'état actuel")
    parser.add_argument("--reset",   action="store_true",   help="Remettre le compteur à zéro")
    args = parser.parse_args()

    state = load_state()

    if args.reset:
        save_state({"sent": [], "total": 0})
        log.info("Compteur remis à zéro.")
        return

    if args.status:
        log.info(f"CVs déjà envoyés : {state['total']}")
        log.info(f"Fichiers : {', '.join(state['sent'][-5:])}{'...' if len(state['sent']) > 5 else ''}")
        return

    if not TESTMAIL_API_KEY:
        log.error("TESTMAIL_API_KEY manquant dans .env")
        sys.exit(1)

    # CVs disponibles non encore envoyés
    all_cvs   = sorted(RAW_DIR.glob("*.txt"))
    already   = set(state["sent"])
    remaining = [f for f in all_cvs if f.name not in already]

    if not remaining:
        log.info("Tous les CVs ont déjà été envoyés.")
        return

    to_send = remaining[:args.n]
    log.info(f"{'[DRY-RUN] ' if args.dry_run else ''}Envoi de {len(to_send)} CVs → {ATS_INBOX}")
    log.info(f"Déjà envoyés : {state['total']} | Restants : {len(remaining)}")
    log.info("")

    sent = errors = 0
    for i, cv in enumerate(to_send, 1):
        log.info(f"  [{i}/{len(to_send)}] {cv.name}")
        if args.dry_run:
            sent += 1
            continue

        if send_cv(cv):
            state["sent"].append(cv.name)
            state["total"] += 1
            sent += 1
        else:
            errors += 1

        if i < len(to_send):
            time.sleep(3)

    if not args.dry_run:
        save_state(state)

    log.info("")
    log.info(f"{'Simulé' if args.dry_run else 'Envoyé'} : {sent} | Erreurs : {errors}")
    log.info(f"Total cumulé : {state['total']} CVs dans l'ATS")
    if not args.dry_run and sent > 0:
        log.info(f"n8n traite les emails toutes les 2 min → visible dans l'ATS dans ~{sent * 2} min")


if __name__ == "__main__":
    main()
