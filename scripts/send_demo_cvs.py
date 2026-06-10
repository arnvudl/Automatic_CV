#!/usr/bin/env python3
"""
send_demo_cvs.py
python scripts/send_demo_cvs.py --n 35
─────────────────
Envoie un batch de CVs par jour directement vers l'API /score.
Garde un historique pour ne jamais renvoyer le même CV.

Usage :
    python scripts/send_demo_cvs.py           # envoie 15 CVs
    python scripts/send_demo_cvs.py --n 20    # envoie 20 CVs
    python scripts/send_demo_cvs.py --dry-run # aperçu sans envoi
    python scripts/send_demo_cvs.py --status  # voir combien ont été envoyés

Exemple pour arriver à 75 en 5 jours :
    Jour 1 (6 juin)  : python scripts/send_demo_cvs.py --n 15
    Jour 2 (7 juin)  : python scripts/send_demo_cvs.py --n 15
    ...
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
API_URL    = os.getenv("ATS_API_URL", "https://api.lony.app")
SCORE_URL  = f"{API_URL}/score"

RAW_DIR    = Path(__file__).parent.parent / "data" / "raw"
STATE_FILE = Path(__file__).parent.parent / "data" / "demo_sent.json"


# ── Tracking ─────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"sent": [], "total": 0}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Envoi ────────────────────────────────────────────────────────────

def send_cv(cv_path: Path) -> bool:
    try:
        with cv_path.open("rb") as f:
            r = requests.post(
                SCORE_URL,
                files={"file": (cv_path.name, f, "text/plain")},
                timeout=30,
            )
        r.raise_for_status()
        result = r.json()
        name     = result.get("name", "?")
        decision = result.get("decision", "?")
        score    = round((result.get("score") or 0) * 100)
        log.info(f"     → {name} | {decision} | {score}%")
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

    # CVs disponibles non encore envoyés
    all_cvs   = sorted(RAW_DIR.glob("*.txt"))
    already   = set(state["sent"])
    remaining = [f for f in all_cvs if f.name not in already]

    if not remaining:
        log.info("Tous les CVs ont déjà été envoyés.")
        return

    to_send = remaining[:args.n]
    log.info(f"{'[DRY-RUN] ' if args.dry_run else ''}Envoi de {len(to_send)} CVs → {SCORE_URL}")
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
        log.info(f"Candidats scorés et visibles immédiatement dans l'ATS.")


if __name__ == "__main__":
    main()
