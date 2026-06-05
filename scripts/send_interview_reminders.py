#!/usr/bin/env python3
"""
send_interview_reminders.py
────────────────────────────
Envoie un email de rappel aux candidats qui ont un entretien aujourd'hui.

Usage :
    python scripts/send_interview_reminders.py

Cron (tous les jours à 8h) :
    0 8 * * * /usr/bin/python3 /root/Automatic_CV/scripts/send_interview_reminders.py >> /var/log/reminders.log 2>&1

Variables d'environnement requises (fichier .env ou export) :
    SMTP_HOST       ex: smtp.gmail.com
    SMTP_PORT       ex: 587  (TLS) ou 465 (SSL)
    SMTP_USER       ex: recrutement@lony.app
    SMTP_PASSWORD   ex: mot de passe ou app password Gmail
    SMTP_FROM       ex: "Luminary RH <recrutement@lony.app>"  (optionnel, = SMTP_USER si absent)
    DATABASE_URL    ex: postgresql://cv_user:cv_pass@localhost:5432/cv_intelligence
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


# ── Main ─────────────────────────────────────────────────────────────

def main():
    if not SMTP_USER or not SMTP_PASSWORD:
        log.error("SMTP_USER et SMTP_PASSWORD doivent être définis dans .env")
        sys.exit(1)

    today = date.today().strftime("%Y-%m-%d")
    log.info(f"Recherche des entretiens du {today}…")

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                i.interview_id,
                i.candidate_id,
                i.candidate_name,
                i.time,
                i.interview_type,
                i.notes,
                c.email,
                c.name AS c_name
            FROM interviews i
            LEFT JOIN candidates c ON c.candidate_id = i.candidate_id
            WHERE i.date = :today
        """), {"today": today}).fetchall()

    if not rows:
        log.info("Aucun entretien aujourd'hui.")
        return

    log.info(f"{len(rows)} entretien(s) trouvé(s).")
    sent = errors = skipped = 0

    for row in rows:
        email         = row.email
        name          = row.candidate_name or row.c_name or "Candidat"
        interview_type = row.interview_type or ""
        interview_time = row.time or ""
        notes         = row.notes or ""

        if not email:
            log.warning(f"  ⚠ Pas d'email pour {name} (id={row.candidate_id}) — ignoré")
            skipped += 1
            continue

        log.info(f"  → {name} <{email}> — {interview_type or 'Entretien'} à {interview_time or '?'}")
        ok = send_email(email, name, interview_type, interview_time, notes)
        if ok:
            sent += 1
        else:
            errors += 1

    log.info(f"Terminé : {sent} envoyé(s), {errors} erreur(s), {skipped} ignoré(s).")


if __name__ == "__main__":
    main()
