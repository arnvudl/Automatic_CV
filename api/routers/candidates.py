"""
routers/candidates.py — Endpoints CRUD candidats.
"""

import re as _re
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import or_, desc as _desc

from api.auth import get_current_user
from api.config import CANDIDATES_FILE, RAW_DIR, RAW_TEXTS_DIR, FEATURE_LABELS
from api.database import get_db, Candidate as CandidateModel
from api.sse import broadcast
from api.scoring import groq_client, GROQ_MODEL

router = APIRouter(tags=["candidates"], dependencies=[Depends(get_current_user)])
logger = logging.getLogger("cv_api")

# ── Regex parsing CV brut ────────────────────────────────────────────
_RE_SUMMARY  = _re.compile(r"Professional Summary:\s*\n(.*?)(?:\n\n|\nEducation:|\nExperience:)", _re.DOTALL)
_RE_TECH     = _re.compile(r"Technical:\s*(.*)", _re.IGNORECASE)
_RE_METH     = _re.compile(r"Methods:\s*(.*)",   _re.IGNORECASE)
_RE_MAN      = _re.compile(r"Management:\s*(.*)", _re.IGNORECASE)
_RE_LANG_BLK = _re.compile(r"Languages:(.*?)(?:Certifications:|$)", _re.DOTALL | _re.IGNORECASE)
_RE_CERT_BLK = _re.compile(r"Certifications:(.*?)$", _re.DOTALL | _re.IGNORECASE)


def _parse_cv_detail(source_filename: str) -> dict:
    """Lit le CV brut et extrait résumé, skills, langues, certifs."""
    candidates_paths = (
        list(RAW_DIR.glob(f"{source_filename}")) +
        list(RAW_DIR.glob(f"{source_filename}.txt")) +
        list(RAW_DIR.glob(f"*{source_filename.replace('.txt','')}*.txt"))
    )
    if not candidates_paths:
        return {}
    text = candidates_paths[0].read_text(encoding="utf-8", errors="replace")

    summary_m = _RE_SUMMARY.search(text)
    summary   = summary_m.group(1).strip() if summary_m else ""

    def _split(pattern, t):
        m = pattern.search(t)
        return [s.strip() for s in m.group(1).split(",") if s.strip()] if m else []

    tech = _split(_RE_TECH, text)
    meth = _split(_RE_METH, text)
    mgmt = _split(_RE_MAN,  text)

    langs = []
    lang_m = _RE_LANG_BLK.search(text)
    if lang_m:
        langs = [l.strip() for l in lang_m.group(1).strip().splitlines() if l.strip()]

    certs = []
    cert_m = _RE_CERT_BLK.search(text)
    if cert_m:
        certs = [
            l.strip() for l in cert_m.group(1).strip().splitlines()
            if l.strip() and l.strip().lower() != "none listed"
        ]

    return {
        "summary":        summary,
        "skills_tech":    tech,
        "skills_meth":    meth,
        "skills_mgmt":    mgmt,
        "languages":      langs,
        "certifications": certs,
    }


# ── GET /candidates ──────────────────────────────────────────────────
@router.get("/candidates")
def list_candidates(
    decision:  Optional[str] = Query(None),
    sector:    Optional[str] = Query(None),
    status:    Optional[str] = Query(None),
    min_score: float         = Query(0.0),
    limit:     int           = Query(200),
    q:         Optional[str] = Query(None, description="recherche nom / email / rôle"),
):
    try:
        with get_db() as db:
            qry = db.query(CandidateModel)
            if decision:
                qry = qry.filter(CandidateModel.decision == decision)
            if sector:
                qry = qry.filter(CandidateModel.sector.ilike(sector))
            if status:
                qry = qry.filter(CandidateModel.status == status)
            if q:
                like = f"%{q}%"
                qry = qry.filter(or_(
                    CandidateModel.name.ilike(like),
                    CandidateModel.email.ilike(like),
                    CandidateModel.target_role.ilike(like),
                    CandidateModel.sector.ilike(like),
                ))
            qry = qry.filter(CandidateModel.score >= min_score)
            rows = qry.order_by(_desc(CandidateModel.score)).limit(limit).all()
            if rows:
                return [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in rows]
    except Exception as db_err:
        logger.warning(f"DB read failed ({db_err}), CSV fallback")

    # CSV fallback
    if not CANDIDATES_FILE.exists():
        return []
    df = pd.read_csv(CANDIDATES_FILE)
    if decision:
        df = df[df["decision"] == decision]
    if sector:
        df = df[df["sector"].str.lower() == sector.lower()]
    if status and "status" in df.columns:
        df = df[df["status"] == status]
    df = df[pd.to_numeric(df["score"], errors="coerce").fillna(0) >= min_score]
    df = df.sort_values("score", ascending=False).head(limit)
    return df.fillna("").to_dict(orient="records")


# ── GET /candidates/{id} ─────────────────────────────────────────────
@router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: str):
    # DB first
    try:
        with get_db() as db:
            obj = db.get(CandidateModel, candidate_id)
            if obj:
                data = {c.name: getattr(obj, c.name) for c in obj.__table__.columns}
                detail = _parse_cv_detail(str(data.get("source_filename", "")))
                return {**data, **detail}
    except Exception as db_err:
        logger.warning(f"DB read failed for candidate {candidate_id}: {db_err}")

    # CSV fallback
    if not CANDIDATES_FILE.exists():
        raise HTTPException(404, "Aucun candidat enregistré.")
    df = pd.read_csv(CANDIDATES_FILE)
    row = df[df["candidate_id"] == candidate_id]
    if row.empty:
        raise HTTPException(404, f"Candidat {candidate_id} introuvable.")
    data = row.iloc[0].fillna("").to_dict()
    detail = _parse_cv_detail(str(data.get("source_filename", "")))
    return {**data, **detail}


# ── PATCH /candidates/{id}/status ────────────────────────────────────
class StatusUpdate(BaseModel):
    status: str

VALID_STATUSES = {"inbox", "review", "interview", "rejected", "archived", "interview_scheduled"}

@router.patch("/candidates/{candidate_id}/status")
async def update_candidate_status(candidate_id: str, body: StatusUpdate):
    if body.status not in VALID_STATUSES:
        raise HTTPException(400, f"Statut invalide. Valeurs acceptées : {VALID_STATUSES}")

    db_ok = False
    try:
        with get_db() as db:
            obj = db.get(CandidateModel, candidate_id)
            if obj:
                obj.status = body.status
                db_ok = True
    except Exception as db_err:
        logger.warning(f"DB status update failed ({db_err})")

    if CANDIDATES_FILE.exists():
        try:
            df = pd.read_csv(CANDIDATES_FILE)
            if candidate_id in df["candidate_id"].values:
                if "status" not in df.columns:
                    df["status"] = "inbox"
                df.loc[df["candidate_id"] == candidate_id, "status"] = body.status
                df.to_csv(CANDIDATES_FILE, index=False)
        except Exception:
            pass

    if not db_ok and not CANDIDATES_FILE.exists():
        raise HTTPException(404, f"Candidat {candidate_id} introuvable.")

    await broadcast({"type": "status_updated",
                     "data": {"candidate_id": candidate_id, "status": body.status}})
    return {"candidate_id": candidate_id, "status": body.status}


# ── DELETE /candidates/{id} ──────────────────────────────────────────
@router.delete("/candidates/{candidate_id}")
async def delete_candidate(candidate_id: str):
    deleted = False
    try:
        with get_db() as db:
            obj = db.get(CandidateModel, candidate_id)
            if obj:
                db.delete(obj)
                deleted = True
    except Exception as db_err:
        logger.warning(f"DB delete failed ({db_err})")

    if CANDIDATES_FILE.exists():
        try:
            df = pd.read_csv(CANDIDATES_FILE)
            if candidate_id in df["candidate_id"].values:
                df = df[df["candidate_id"] != candidate_id]
                df.to_csv(CANDIDATES_FILE, index=False)
                deleted = True
        except Exception:
            pass

    if not deleted:
        raise HTTPException(404, f"Candidat {candidate_id} introuvable.")

    await broadcast({"type": "candidate_deleted", "data": {"candidate_id": candidate_id}})
    return {"deleted": True, "candidate_id": candidate_id}


# ── GET /candidates/{id}/explain ─────────────────────────────────────
@router.get("/candidates/{candidate_id}/explain")
def explain_candidate(candidate_id: str):
    import json
    if not CANDIDATES_FILE.exists():
        raise HTTPException(404, "Aucun candidat enregistré.")
    df = pd.read_csv(CANDIDATES_FILE)
    row = df[df["candidate_id"] == candidate_id]
    if row.empty:
        raise HTTPException(404, f"Candidat {candidate_id} introuvable.")
    r = row.iloc[0]
    try:
        sv = json.loads(r.get("shap_json") or "{}")
    except Exception:
        sv = {}
    score = float(r.get("score") or 0)
    thr   = float(r.get("threshold_used") or 0.5)

    invited   = df[df["decision"] == "invite"]
    feat_cols = [c for c in FEATURE_LABELS if c in df.columns]
    missing   = []
    for c in feat_cols:
        candidate_val = float(pd.to_numeric(r.get(c), errors="coerce") or 0)
        invited_avg   = float(pd.to_numeric(invited[c], errors="coerce").mean() or 0)
        if invited_avg > 0 and candidate_val < invited_avg * 0.7:
            missing.append({
                "feature":         FEATURE_LABELS.get(c, c),
                "candidate_value": round(candidate_val, 3),
                "invited_avg":     round(invited_avg, 3),
                "gap_pct":         round((invited_avg - candidate_val) / invited_avg * 100, 1),
            })

    from api.routers.stats import shap_narrative
    labeled_shap = {FEATURE_LABELS.get(k, k): v for k, v in sv.items()}
    return {
        "shap":      labeled_shap,
        "narrative": shap_narrative(sv, score, thr),
        "missing":   sorted(missing, key=lambda x: x["gap_pct"], reverse=True)[:4],
        "score":     score,
        "threshold": thr,
    }


# ── GET /candidates/{id}/semantic ────────────────────────────────────
@router.get("/candidates/{candidate_id}/semantic")
def semantic_analysis(candidate_id: str):
    import json, re
    if groq_client is None:
        raise HTTPException(503, "Client Groq non initialisé. Vérifiez GROQ_API_KEY.")

    raw_file = RAW_TEXTS_DIR / f"{candidate_id}.txt"
    if not raw_file.exists():
        raise HTTPException(404, f"Texte CV introuvable pour {candidate_id}.")

    cv_text = raw_file.read_text(encoding="utf-8", errors="ignore")[:12000]

    context_snippet = ""
    if CANDIDATES_FILE.exists():
        df = pd.read_csv(CANDIDATES_FILE)
        row = df[df["candidate_id"] == candidate_id]
        if not row.empty:
            r = row.iloc[0]
            context_snippet = (
                f"Score ML : {r.get('score', '?')} | Décision : {r.get('decision', '?')} | "
                f"Expérience : {r.get('years_experience', '?')} ans | "
                f"Secteur : {r.get('sector', '?')} | Éducation : {r.get('education_level', '?')}"
            )

    prompt = f"""Tu es un expert RH senior et psychologue du travail. Analyse ce CV de façon approfondie et bienveillante.

CONTEXTE ML : {context_snippet}

CV BRUT :
\"\"\"
{cv_text}
\"\"\"

Réponds UNIQUEMENT en JSON valide avec cette structure :
{{
  "semantic_score": <entier 0-100>,
  "trajectory": "<2-3 phrases>",
  "skill_equivalencies": [{{"stated": "...", "equivalent": "...", "level": "junior|confirmed|expert"}}],
  "career_gaps": [{{"period": "...", "likely_reason": "...", "impact": "faible|modéré|fort"}}],
  "hidden_gems": ["..."],
  "red_flags": ["..."],
  "recommendation": "<une phrase>"
}}

Limite skill_equivalencies à 5, hidden_gems à 3, red_flags à 2."""

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL, max_tokens=1024, temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = completion.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(json_match.group() if json_match else raw)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Réponse LLM non parseable : {e}")
    except Exception as e:
        raise HTTPException(500, f"Erreur Groq API : {e}")

    return {"candidate_id": candidate_id, "model": GROQ_MODEL, **result}
