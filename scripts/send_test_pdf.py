#!/usr/bin/env python3
"""
send_test_pdf.py
─────────────────
Génère un CV fictif au format PDF et l'envoie par email à
73cn1.rh@inbox.testmail.app pour tester le workflow n8n end-to-end.

Usage :
    python scripts/send_test_pdf.py
    python scripts/send_test_pdf.py --role "Data Analyst"
    python scripts/send_test_pdf.py --decision invite   # forcer le profil fort
    python scripts/send_test_pdf.py --decision reject   # forcer le profil faible
"""

import argparse
import base64
import json
import random
import sys
import textwrap
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────
TESTMAIL_APIKEY   = "6d3a712a-6383-4888-97d1-091d024142e8"
TESTMAIL_NS       = "73cn1"
TESTMAIL_SEND_URL = f"https://api.testmail.app/api/send?apikey={TESTMAIL_APIKEY}"
INBOX             = f"{TESTMAIL_NS}.rh@inbox.testmail.app"

# Profils de test
PROFILES = {
    "invite": {
        "name": "Sophie Marchand",
        "email": "sophie.marchand@gmail.com",
        "role": "Data Analyst",
        "experience": "5 ans",
        "skills": "Python, SQL, Power BI, Tableau, Excel, statistiques descriptives",
        "education": "Master en Statistiques appliquées — Université Paris-Saclay",
        "exp_text": (
            "Data Analyst chez BNP Paribas (3 ans) : création de dashboards Power BI, "
            "analyse de portefeuille, requêtes SQL complexes sur Oracle.\n"
            "Business Analyst chez Capgemini (2 ans) : modélisation de données, "
            "reporting automatisé, coordination avec les équipes métier."
        ),
    },
    "reject": {
        "name": "Kevin Durand",
        "email": "kevin.durand@hotmail.fr",
        "role": "Assistant administratif",
        "experience": "1 an",
        "skills": "Word, Excel basique, accueil téléphonique",
        "education": "Bac pro Secrétariat",
        "exp_text": (
            "Assistant administratif dans une PME (1 an) : gestion du courrier, "
            "classement de documents, accueil des visiteurs."
        ),
    },
}


def build_cv_text(profile: dict) -> str:
    return textwrap.dedent(f"""
        CURRICULUM VITAE
        ================

        {profile['name']}
        Email  : {profile['email']}
        Poste  : {profile['role']}

        FORMATION
        ---------
        {profile['education']}

        EXPÉRIENCE PROFESSIONNELLE ({profile['experience']})
        --------------------------------
        {profile['exp_text']}

        COMPÉTENCES TECHNIQUES
        ----------------------
        {profile['skills']}
    """).strip()


def make_pdf(cv_text: str, out_path: Path) -> Path:
    """Génère un PDF minimal sans dépendance externe (pure Python)."""
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        for line in cv_text.split("\n"):
            pdf.cell(0, 6, line, ln=True)
        pdf.output(str(out_path))
        return out_path
    except ImportError:
        pass

    # Fallback : PDF minimal hand-crafted (ne nécessite aucune lib)
    lines = cv_text.split("\n")
    # On utilise reportlab si disponible
    try:
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(str(out_path))
        y = 800
        for line in lines:
            c.drawString(50, y, line)
            y -= 14
            if y < 50:
                c.showPage()
                y = 800
        c.save()
        return out_path
    except ImportError:
        pass

    # Dernier recours : PDF ultra-minimal structuré à la main
    # (valide, lisible par tous les readers)
    stream_content = "\n".join(f"BT /F1 11 Tf 50 {800 - i*14} Td ({_pdf_escape(l)}) Tj ET"
                               for i, l in enumerate(lines) if i < 50)
    stream_bytes = stream_content.encode("latin-1", errors="replace")
    stream_len = len(stream_bytes)

    body = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842]\n"
        b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        + f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode()
        + stream_bytes
        + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )
    out_path.write_bytes(body)
    return out_path


def _pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def send_email(profile: dict, pdf_path: Path):
    """
    testmail.app n'expose pas de SMTP entrant et l'API send ne supporte
    pas les pièces jointes. On sauvegarde le PDF dans le dossier courant
    et on affiche les instructions pour l'envoi manuel.
    """
    dest = Path(f"cv_test_{profile['role'].replace(' ', '_')}.pdf")
    import shutil
    shutil.copy(pdf_path, dest)
    print(f"\n    PDF sauvegarde : {dest.resolve()}")
    print( "")
    print( "    Pour tester le workflow n8n :")
    print(f"    1. Ouvre ta boite mail (Gmail, Outlook...)")
    print(f"    2. Compose un nouvel email")
    print(f"    3. Destinataire : {INBOX}")
    print(f"    4. Objet : Candidature {profile['role']}")
    print(f"    5. Attache le fichier : {dest.resolve()}")
    print( "    6. Envoie !")
    print( "")
    print( "    Le workflow n8n est actif et tourne toutes les 2 min.")
    print( "    Verifie l'execution : https://n8n.lony.app/workflow/mSyTvS1vUhw4Jrs7")
    return {"status": "pdf_saved", "path": str(dest.resolve())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role",     default=None, help="Titre du poste (optionnel)")
    parser.add_argument("--decision", choices=["invite", "reject"], default=None,
                        help="Profil à envoyer (invite=fort, reject=faible). Par défaut: aléatoire.")
    args = parser.parse_args()

    decision = args.decision or random.choice(["invite", "reject"])
    profile  = dict(PROFILES[decision])
    if args.role:
        profile["role"] = args.role

    print(f"\n[PDF]  Generation du CV — {profile['name']} ({profile['role']})")
    cv_text  = build_cv_text(profile)
    tmp_path = Path("cv_test.pdf")
    make_pdf(cv_text, tmp_path)
    print(f"    Fichier : {tmp_path} ({tmp_path.stat().st_size} octets)")

    send_email(profile, tmp_path)
    tmp_path.unlink(missing_ok=True)
    print(f"\n[INFO] Profil attendu : {decision.upper()} - score {'eleve -> invite' if decision == 'invite' else 'faible -> reject'}\n")


if __name__ == "__main__":
    main()
