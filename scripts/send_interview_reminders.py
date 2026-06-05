#!/usr/bin/env python3
"""
send_interview_reminders.py
────────────────────────────
Envoie un email de rappel aux candidats qui ont un entretien aujourd'hui.

Sources (par ordre de priorité) :
  1. data/interviews_schedule.csv  — planning manuel (prioritaire)
  2. Base de données PostgreSQL/SQLite — table interviews JOIN candidates

Usage :
    python scripts/send_interview_reminders.py           # mode normal
    python scripts/send_interview_reminders.py --dry-run # aperçu sans envoi
    python scripts/send_interview_reminders.py --date 2026-06-10  # forcer une date

Cron pour le 10 juin 2026 (présentation) :
    0 8 10 6 * /usr/bin/python3 /root/Automatic_CV/scripts/send_interview_reminders.py >> /var/log/reminders.log 2>&1

Cron quotidien (production) :
    0 8 * * * /usr/bin/python3 /root/Automatic_CV/scripts/send_interview_reminders.py >> /var/log/reminders.log 2>&1

Variables d'environnement requises (fichier .env ou export) :
    SMTP_HOST       ex: smtp.gmail.com
    SMTP_PORT       ex: 587  (TLS) ou 465 (SSL)
    SMTP_USER       ex: recrutement@lony.app
    SMTP_PASSWORD   ex: mot de passe ou app password Gmail
    SMTP_FROM       ex: "Luminary RH <recrutement@lony.app>"  (optionnel, = SMTP_USER si absent)
    DATABASE_URL    ex: postgresql://cv_user:cv_pass@localhost:5432/cv_intelligence

Format data/interviews_schedule.csv :
    name,email,date,time,interview_type,notes
    Jean Dupont,jean@example.com,2026-06-10,10:00,Entretien technique,Focus Python/SQL
"""

import os
import sys
import smtplib
import logging
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# ── Chargement .env ─────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("reminders")

# ── Config SMTP ─────────────────────────────────────────────────────
SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM     = os.getenv("SMTP_FROM", SMTP_USER)

# ── DB ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./data/cv_intelligence.db",
)

from sqlalchemy import create_engine, text
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)


# ── Template email ───────────────────────────────────────────────────

def build_email_html(candidate_name: str, interview_type: str, interview_time: str, notes: str) -> str:
    time_display = interview_time or "heure à confirmer"
    type_display = interview_type or "Entretien"
    notes_section = f"""
        <tr>
          <td style="padding:16px 32px 0">
            <p style="margin:0;font-size:13px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.06em">Notes</p>
            <p style="margin:8px 0 0;font-size:14px;color:#1e293b;line-height:1.6">{notes}</p>
          </td>
        </tr>
    """ if notes and notes.strip() else ""

    return f"""<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:'Segoe UI',system-ui,sans-serif">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;padding:40px 16px">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,.08)">

        <!-- Header -->
        <tr>
          <td style="background:linear-gradient(135deg,#6366f1,#8b5cf6);padding:40px 32px;text-align:center">
            <p style="margin:0;font-size:13px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#c4b5fd">Luminary ATS</p>
            <h1 style="margin:12px 0 0;font-size:26px;font-weight:800;color:#ffffff;line-height:1.2">
              Rappel d'entretien
            </h1>
          </td>
        </tr>

        <!-- Body -->
        <tr>
          <td style="padding:32px 32px 0">
            <p style="margin:0;font-size:16px;color:#1e293b;line-height:1.6">
              Bonjour <strong>{candidate_name}</strong>,
            </p>
            <p style="margin:16px 0 0;font-size:15px;color:#475569;line-height:1.7">
              Nous vous rappelons que votre entretien est prévu <strong>aujourd'hui</strong>.
              Voici les détails :
            </p>
          </td>
        </tr>

        <!-- Info box -->
        <tr>
          <td style="padding:24px 32px">
            <table width="100%" cellpadding="0" cellspacing="0"
              style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden">
              <tr>
                <td style="padding:16px 24px;border-bottom:1px solid #e2e8f0">
                  <p style="margin:0;font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.08em">Type d'entretien</p>
                  <p style="margin:6px 0 0;font-size:16px;font-weight:700;color:#6366f1">{type_display}</p>
                </td>
              </tr>
              <tr>
                <td style="padding:16px 24px">
                  <p style="margin:0;font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.08em">Heure</p>
                  <p style="margin:6px 0 0;font-size:16px;font-weight:700;color:#1e293b">{time_display}</p>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        {notes_section}

        <!-- CTA -->
        <tr>
          <td style="padding:24px 32px 40px;text-align:center">
            <p style="margin:0 0 16px;font-size:14px;color:#64748b;line-height:1.6">
              Pour toute question ou modification, n'hésitez pas à nous contacter en répondant à cet email.
            </p>
            <p style="margin:0;font-size:14px;color:#94a3b8">
              Bonne chance !<br>
              <strong style="color:#1e293b">L'équipe RH — Luminary</strong>
            </p>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:#f8fafc;border-top:1px solid #e2e8f0;padding:16px 32px;text-align:center">
            <p style="margin:0;font-size:12px;color:#94a3b8">
              Cet email a été envoyé automatiquement par Luminary ATS.<br>
              © {date.today().year} Luminary — Tous droits réservés.
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


def build_email_text(candidate_name: str, interview_type: str, interview_time: str, notes: str) -> str:
    time_display = interview_time or "heure à confirmer"
    type_display = interview_type or "Entretien"
    lines = [
        f"Bonjour {candidate_name},",
        "",
        "Nous vous rappelons que votre entretien est prévu AUJOURD'HUI.",
        "",
        f"Type     : {type_display}",
        f"Heure    : {time_display}",
    ]
    if notes and notes.strip():
        lines += ["", f"Notes    : {notes}"]
    lines += [
        "",
        "Pour toute question, répondez à cet email.",
        "",
        "Bonne chance !",
        "L'équipe RH — Luminary",
    ]
    return "\n".join(lines)


# ── Envoi SMTP ───────────────────────────────────────────────────────

def send_email(to: str, candidate_name: str, interview_type: str,
               interview_time: str, notes: str) -> bool:
    """Envoie le rappel à `to`. Retourne True si succès."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Rappel entretien — {date.today().strftime('%d/%m/%Y')}"
    msg["From"]    = SMTP_FROM
    msg["To"]      = to

    msg.attach(MIMEText(
        build_email_text(candidate_name, interview_type, interview_time, notes), "plain", "utf-8"
    ))
    msg.attach(MIMEText(
        build_email_html(candidate_name, interview_type, interview_time, notes), "html", "utf-8"
    ))

    try:
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15) as smtp:
                smtp.login(SMTP_USER, SMTP_PASSWORD)
                smtp.sendmail(SMTP_FROM, [to], msg.as_bytes())
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(SMTP_USER, SMTP_PASSWORD)
                smtp.sendmail(SMTP_FROM, [to], msg.as_bytes())
        log.info(f"  ✓ Email envoyé → {to}")
        return True
    except Exception as e:
        log.error(f"  ✗ Échec envoi → {to} : {e}")
        return False


# ── Lecture sources ──────────────────────────────────────────────────

def load_from_csv(today: str) -> list[dict]:
    """Lit data/interviews_schedule.csv et filtre sur la date du jour."""
    csv_path = Path(__file__).parent.parent / "data" / "interviews_schedule.csv"
    if not csv_path.exists():
        return []
    import csv as _csv
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            if r.get("date", "").strip() == today:
                rows.append({
                    "name":           r.get("name", "").strip(),
                    "email":          r.get("email", "").strip(),
                    "interview_type": r.get("interview_type", "").strip(),
                    "time":           r.get("time", "").strip(),
                    "notes":          r.get("notes", "").strip(),
                    "source":         "CSV",
                })
    return rows


def load_from_db(today: str) -> list[dict]:
    """Lit la table interviews JOIN candidates pour la date du jour."""
    try:
        with engine.connect() as conn:
            raw = conn.execute(text("""
                SELECT i.candidate_name, i.time, i.interview_type, i.notes,
                       c.email, c.name AS c_name
                FROM interviews i
                LEFT JOIN candidates c ON c.candidate_id = i.candidate_id
                WHERE i.date = :today
            """), {"today": today}).fetchall()
        return [{
            "name":           r.candidate_name or r.c_name or "Candidat",
            "email":          r.email or "",
            "interview_type": r.interview_type or "",
            "time":           r.time or "",
            "notes":          r.notes or "",
            "source":         "DB",
        } for r in raw]
    except Exception as e:
        log.warning(f"Lecture DB échouée : {e}")
        return []


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche les emails sans les envoyer")
    parser.add_argument("--date", default=None,
                        help="Forcer une date YYYY-MM-DD (défaut: aujourd'hui)")
    args = parser.parse_args()

    if not SMTP_USER or not SMTP_PASSWORD:
        if not args.dry_run:
            log.error("SMTP_USER et SMTP_PASSWORD doivent être définis dans .env")
            sys.exit(1)
        log.warning("SMTP non configuré — mode dry-run forcé")
        args.dry_run = True

    today = args.date or date.today().strftime("%Y-%m-%d")
    log.info(f"{'[DRY-RUN] ' if args.dry_run else ''}Recherche des entretiens du {today}…")

    # Priorité CSV → DB (le CSV est le planning manuel, la DB est la source opérationnelle)
    rows = load_from_csv(today)
    if rows:
        log.info(f"  Source : CSV ({len(rows)} entrée(s) trouvée(s))")
    else:
        rows = load_from_db(today)
        if rows:
            log.info(f"  Source : DB ({len(rows)} entrée(s) trouvée(s))")

    if not rows:
        log.info("Aucun entretien aujourd'hui.")
        return

    sent = errors = skipped = 0

    for r in rows:
        name           = r["name"] or "Candidat"
        email          = r["email"]
        interview_type = r["interview_type"]
        interview_time = r["time"]
        notes          = r["notes"]

        if not email:
            log.warning(f"  ⚠ Pas d'email pour {name} — ignoré")
            skipped += 1
            continue

        log.info(f"  → {name} <{email}> — {interview_type or 'Entretien'} à {interview_time or '?'} [{r['source']}]")

        if args.dry_run:
            log.info(f"    [DRY-RUN] Email non envoyé")
            sent += 1
            continue

        ok = send_email(email, name, interview_type, interview_time, notes)
        if ok:
            sent += 1
        else:
            errors += 1

    log.info(f"Terminé : {sent} {'simulé(s)' if args.dry_run else 'envoyé(s)'}, {errors} erreur(s), {skipped} ignoré(s).")


if __name__ == "__main__":
    main()
