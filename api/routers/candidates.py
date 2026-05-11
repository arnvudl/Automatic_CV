"""
routers/candidates.py — Endpoints CRUD candidats.
"""

import re as _re
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Depends
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


def _parse_cv_detail(source_filename: str, candidate_id: str = "") -> dict:
    """Lit le CV brut et extrait résumé, skills, langues, certifs.
    Cherche d'abord dans RAW_TEXTS_DIR/{candidate_id}.txt (texte extrait au scoring),
    puis dans RAW_DIR par source_filename (fichiers bruts originaux)."""
    text = ""

    # 1. RAW_TEXTS_DIR/{candidate_id}.txt — source principale
    if candidate_id:
        p = RAW_TEXTS_DIR / f"{candidate_id}.txt"
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="replace")

    # 2. RAW_DIR par source_filename — fallback
    if not text and source_filename:
        candidates_paths = (
            list(RAW_DIR.glob(source_filename)) +
            list(RAW_DIR.glob(f"{source_filename}.txt")) +
            list(RAW_DIR.glob(f"*{source_filename.replace('.txt', '')}*.txt"))
        )
        if candidates_paths:
            text = candidates_paths[0].read_text(encoding="utf-8", errors="replace")

    if not text:
        return {}

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
    db_rows = []
    db_ids  = set()
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
            db_rows = [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in rows]
            db_ids  = {r["candidate_id"] for r in db_rows}
    except Exception as db_err:
        logger.warning(f"DB read failed ({db_err}), CSV fallback")

    # Fusion avec le CSV — candidats absents de la DB
    csv_rows = []
    if CANDIDATES_FILE.exists():
        try:
            df = pd.read_csv(CANDIDATES_FILE, on_bad_lines='skip')
            if decision:
                df = df[df["decision"] == decision]
            if sector:
                df = df[df["sector"].str.lower() == sector.lower()]
            if status and "status" in df.columns:
                df = df[df["status"] == status]
            if q:
                mask = (
                    df["name"].str.contains(q, case=False, na=False) |
                    df.get("target_role", pd.Series(dtype=str)).str.contains(q, case=False, na=False) |
                    df.get("sector",      pd.Series(dtype=str)).str.contains(q, case=False, na=False)
                )
                df = df[mask]
            df = df[pd.to_numeric(df["score"], errors="coerce").fillna(0) >= min_score]
            # Exclure ceux déjà en DB
            df = df[~df["candidate_id"].isin(db_ids)]
            csv_rows = df.fillna("").to_dict(orient="records")
        except Exception as csv_err:
            logger.warning(f"CSV read failed: {csv_err}")

    merged = db_rows + csv_rows
    merged.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
    return merged[:limit]


# ── GET /candidates/{id} ─────────────────────────────────────────────
@router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: str):
    import json as _json

    def _expand_cv_extra(data: dict) -> dict:
        """
        Extrait cv_extra_json de la DB et l'injecte à plat dans data.
        Si cv_extra_json est vide, tente un parse depuis les fichiers bruts
        et sauvegarde le résultat en DB pour les prochaines requêtes.
        """
        cv_extra_raw = data.get("cv_extra_json")
        cv_extra = {}

        if cv_extra_raw and cv_extra_raw not in ("{}", "null", ""):
            try:
                cv_extra = _json.loads(cv_extra_raw)
            except Exception:
                pass

        if not cv_extra:
            # Fallback : parse depuis les fichiers bruts (une seule fois)
            cv_extra = _parse_cv_detail(
                str(data.get("source_filename", "")), candidate_id
            )
            # Sauvegarde en DB pour éviter de relire le fichier la prochaine fois
            if cv_extra:
                try:
                    with get_db() as db2:
                        obj2 = db2.get(CandidateModel, candidate_id)
                        if obj2 and not obj2.cv_extra_json:
                            obj2.cv_extra_json = _json.dumps(cv_extra, ensure_ascii=False)
                except Exception:
                    pass

        return {
            **data,
            "summary":        cv_extra.get("summary", ""),
            "skills_tech":    cv_extra.get("skills_tech", []),
            "skills_meth":    cv_extra.get("skills_meth", []),
            "skills_mgmt":    cv_extra.get("skills_mgmt", []),
            "languages":      cv_extra.get("languages", []),
            "certifications": cv_extra.get("certifications", []),
            "jobs":           cv_extra.get("jobs", []),
            "education":      cv_extra.get("education", []),
        }

    # DB first
    try:
        with get_db() as db:
            obj = db.get(CandidateModel, candidate_id)
            if obj:
                data = {c.name: getattr(obj, c.name) for c in obj.__table__.columns}
                return _expand_cv_extra(data)
    except Exception as db_err:
        logger.warning(f"DB read failed for candidate {candidate_id}: {db_err}")

    # CSV fallback
    if not CANDIDATES_FILE.exists():
        raise HTTPException(404, "Aucun candidat enregistré.")
    df = pd.read_csv(CANDIDATES_FILE, on_bad_lines='skip')
    row = df[df["candidate_id"] == candidate_id]
    if row.empty:
        raise HTTPException(404, f"Candidat {candidate_id} introuvable.")
    data = row.iloc[0].fillna("").to_dict()
    detail = _parse_cv_detail(str(data.get("source_filename", "")), candidate_id)
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
            df = pd.read_csv(CANDIDATES_FILE, on_bad_lines='skip')
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
            df = pd.read_csv(CANDIDATES_FILE, on_bad_lines='skip')
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

    # ── Candidat depuis la DB ────────────────────────────────────────
    candidate_data = None
    try:
        with get_db() as db:
            obj = db.get(CandidateModel, candidate_id)
            if obj:
                candidate_data = {c.name: getattr(obj, c.name) for c in obj.__table__.columns}
    except Exception as e:
        logger.warning(f"DB explain lookup failed: {e}")

    # Fallback CSV robuste (on_bad_lines='skip' ignore les lignes corrompues)
    if candidate_data is None:
        if not CANDIDATES_FILE.exists():
            raise HTTPException(404, "Aucun candidat enregistré.")
        try:
            df_all = pd.read_csv(CANDIDATES_FILE, on_bad_lines='skip')
            row = df_all[df_all["candidate_id"] == candidate_id]
            if row.empty:
                raise HTTPException(404, f"Candidat {candidate_id} introuvable.")
            candidate_data = row.iloc[0].to_dict()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Erreur lecture CSV : {e}")

    try:
        sv = json.loads(candidate_data.get("shap_json") or "{}")
    except Exception:
        sv = {}
    score = float(candidate_data.get("score") or 0)
    thr   = float(candidate_data.get("threshold_used") or 0.5)

    # Calcul des facteurs manquants
    # Seules education_level et years_experience sont en DB ; les autres features
    # (exp_per_year_of_age, etc.) sont des features ML non stockées → on les ignore
    DB_FEATURES = {
        "education_level":   FEATURE_LABELS.get("education_level",   "Niveau études"),
        "years_experience":  FEATURE_LABELS.get("years_experience",  "Expérience"),
    }
    missing = []
    try:
        with get_db() as db:
            invited_rows = db.query(CandidateModel).filter(CandidateModel.decision == "invite").all()
            if invited_rows:
                for feat_key, feat_label in DB_FEATURES.items():
                    candidate_val = float(candidate_data.get(feat_key) or 0)
                    invited_vals  = [
                        float(getattr(r, feat_key))
                        for r in invited_rows
                        if getattr(r, feat_key, None) is not None
                    ]
                    if not invited_vals:
                        continue
                    invited_avg = sum(invited_vals) / len(invited_vals)
                    if invited_avg > 0 and candidate_val < invited_avg * 0.7:
                        missing.append({
                            "feature":         feat_label,
                            "candidate_value": round(candidate_val, 3),
                            "invited_avg":     round(invited_avg, 3),
                            "gap_pct":         round((invited_avg - candidate_val) / invited_avg * 100, 1),
                        })
    except Exception as e:
        logger.warning(f"Missing factors computation failed: {e}")

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
        df = pd.read_csv(CANDIDATES_FILE, on_bad_lines='skip')
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
