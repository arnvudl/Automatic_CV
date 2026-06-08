"""
routers/pipeline.py — Kanban pipeline : étapes par offre + déplacement candidats.

Endpoints :
  GET  /jobs/{job_id}/stages        → étapes du pipeline (crée les defaults si vide)
  POST /jobs/{job_id}/stages        → crée une étape custom
  GET  /jobs/{job_id}/candidates    → liste des candidats assignés à cette offre
  PATCH /candidates/{id}/stage      → déplace un candidat vers une étape
"""

import re
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.auth import get_current_user
from api.database import get_db, PipelineStage as StageModel, Candidate as CandidateModel, Job as JobModel

router = APIRouter(tags=["pipeline"], dependencies=[Depends(get_current_user)])
logger = logging.getLogger("cv_api")

# ── Matching titre de poste ↔ rôle cible candidat ─────────────────────
_STOP_WORDS = {
    "the", "a", "an", "of", "and", "et", "de", "du", "le", "la", "les",
    "senior", "junior", "lead", "sr", "sr.", "jr", "jr.", "i", "ii", "iii", "iv",
}
_RE_WORD = re.compile(r"[a-zàâäéèêëïîôöùûüÿæœç]+")


def _significant_words(title: str) -> list[str]:
    return [w for w in _RE_WORD.findall(title.lower()) if len(w) > 2 and w not in _STOP_WORDS]


def _title_matches(job_title: Optional[str], candidate_role: Optional[str]) -> bool:
    """Heuristique 'regex-like' : tous les mots significatifs du titre de poste
    doivent apparaître (par préfixe, façon stemming léger) dans le rôle cible
    du candidat. Évite par ex. qu'une offre 'Data Analyst' affiche un
    'Data Scientist' (mot 'data' commun mais 'analyst' absent)."""
    if not job_title or not candidate_role:
        return False
    words = _significant_words(job_title)
    cr = candidate_role.lower()
    if not words:
        return job_title.lower().strip() in cr
    return all(re.search(re.escape(w[:5]), cr) for w in words)


@router.get("/candidates/meta/target-roles")
def get_target_roles():
    """Retourne la liste des rôles cibles distincts présents dans les CVs —
    utilisée pour l'autocomplétion du champ 'Titre du poste' à la création d'une offre."""
    try:
        with get_db() as db:
            rows = (
                db.query(CandidateModel.target_role)
                .filter(CandidateModel.target_role.isnot(None))
                .distinct()
                .all()
            )
            roles = sorted({r[0].strip() for r in rows if r[0] and r[0].strip()})
            return roles
    except Exception as e:
        logger.error(f"Error getting target roles: {e}")
        return []

# ── Étapes par défaut ─────────────────────────────────────────────────
DEFAULT_STAGES = [
    {"key": "inbox",               "name": "Inbox",                  "position": 0, "color": "#6b7280"},
    {"key": "screening",           "name": "Screening",              "position": 1, "color": "#3b82f6"},
    {"key": "phone_interview",     "name": "Entretien téléph.",      "position": 2, "color": "#8b5cf6"},
    {"key": "technical_interview", "name": "Entretien technique",    "position": 3, "color": "#6366f1"},
    {"key": "offer_sent",          "name": "Offre envoyée",          "position": 4, "color": "#f59e0b"},
    {"key": "hired",               "name": "Embauché ✓",             "position": 5, "color": "#10b981"},
    {"key": "rejected",            "name": "Refusé",                 "position": 6, "color": "#ef4444"},
]


# ── Helpers ───────────────────────────────────────────────────────────
def _stage_to_dict(s: StageModel, candidates: list[dict]) -> dict:
    return {
        "stage_id":   s.stage_id,
        "job_id":     s.job_id,
        "name":       s.name,
        "position":   s.position,
        "color":      s.color,
        "candidates": [c for c in candidates if c["stage_id"] == s.stage_id],
    }


def _candidate_brief(c: CandidateModel) -> dict:
    return {
        "candidate_id":    c.candidate_id,
        "name":            c.name,
        "sector":          c.sector,
        "target_role":     c.target_role,
        "score":           c.score,
        "decision":        c.decision,
        "status":          c.status,
        "stage_id":        c.stage_id,
        "years_experience": c.years_experience,
        "received_at":     c.received_at.isoformat() if c.received_at else None,
    }


# ── Pydantic ──────────────────────────────────────────────────────────
class StageCreate(BaseModel):
    name:     str
    position: Optional[int] = None
    color:    Optional[str] = "#6b7280"


class StagePatch(BaseModel):
    stage_id: Optional[str]   # None = retirer du pipeline


# ── GET /jobs/{job_id}/stages ─────────────────────────────────────────
@router.get("/jobs/{job_id}/stages")
def get_stages(job_id: str):
    """Retourne les étapes du pipeline pour une offre.
    Crée les étapes par défaut si aucune n'existe encore."""
    try:
        with get_db() as db:
            stages = (
                db.query(StageModel)
                .filter(StageModel.job_id == job_id)
                .order_by(StageModel.position)
                .all()
            )

            # Création des étapes par défaut si vide
            if not stages:
                for s in DEFAULT_STAGES:
                    stage = StageModel(
                        stage_id=f"{job_id}_{s['key']}",
                        job_id=job_id,
                        name=s["name"],
                        position=s["position"],
                        color=s["color"],
                    )
                    db.add(stage)
                db.flush()
                stages = (
                    db.query(StageModel)
                    .filter(StageModel.job_id == job_id)
                    .order_by(StageModel.position)
                    .all()
                )

            # Candidats explicitement assignés à ce job
            valid_stage_ids = {s.stage_id for s in stages}
            assigned = (
                db.query(CandidateModel)
                .filter(CandidateModel.stage_id.in_(valid_stage_ids))
                .all()
            )
            cand_briefs = [_candidate_brief(c) for c in assigned]

            # La première colonne (Inbox) accueille aussi automatiquement
            # les candidats scorés mais pas encore assignés à un pipeline —
            # prêts à être glissés dans les étapes suivantes. On ne montre
            # que ceux dont le rôle cible correspond au titre de l'offre,
            # pour éviter de mélanger des profils de métiers différents.
            inbox_stage = min(stages, key=lambda s: s.position) if stages else None
            if inbox_stage:
                job = db.query(JobModel).filter(JobModel.job_id == job_id).first()
                job_title = job.title if job else None
                unassigned = (
                    db.query(CandidateModel)
                    .filter(CandidateModel.stage_id.is_(None))
                    .filter(CandidateModel.decision != "eliminated")
                    .order_by(CandidateModel.received_at.desc())
                    .limit(200)
                    .all()
                )
                matching = [c for c in unassigned if _title_matches(job_title, c.target_role)]
                cand_briefs += [
                    {**_candidate_brief(c), "stage_id": inbox_stage.stage_id, "unassigned": True}
                    for c in matching[:100]
                ]

            return [_stage_to_dict(s, cand_briefs) for s in stages]

    except Exception as e:
        logger.error(f"Error getting stages for job {job_id}: {e}")
        raise HTTPException(500, "Erreur lors de la récupération des étapes.")


# ── GET /jobs/{job_id}/candidates ─────────────────────────────────────
@router.get("/jobs/{job_id}/candidates")
def get_job_candidates(job_id: str):
    """Retourne les candidats explicitement assignés à cette offre
    (via leur stage_id préfixé par le job_id), avec le nom de leur étape."""
    try:
        with get_db() as db:
            stages = (
                db.query(StageModel)
                .filter(StageModel.job_id == job_id)
                .order_by(StageModel.position)
                .all()
            )
            stage_names = {s.stage_id: s.name for s in stages}
            valid_stage_ids = list(stage_names.keys())
            if not valid_stage_ids:
                return []

            assigned = (
                db.query(CandidateModel)
                .filter(CandidateModel.stage_id.in_(valid_stage_ids))
                .order_by(CandidateModel.score.desc())
                .all()
            )
            return [
                {**_candidate_brief(c), "stage_name": stage_names.get(c.stage_id)}
                for c in assigned
            ]
    except Exception as e:
        logger.error(f"Error getting candidates for job {job_id}: {e}")
        raise HTTPException(500, "Erreur lors de la récupération des candidats.")


# ── POST /jobs/{job_id}/stages ────────────────────────────────────────
@router.post("/jobs/{job_id}/stages", status_code=201)
def create_stage(job_id: str, body: StageCreate):
    """Crée une étape custom pour une offre."""
    try:
        with get_db() as db:
            # Position = après la dernière
            count = db.query(StageModel).filter(StageModel.job_id == job_id).count()
            stage = StageModel(
                stage_id=uuid.uuid4().hex,
                job_id=job_id,
                name=body.name,
                position=body.position if body.position is not None else count,
                color=body.color or "#6b7280",
            )
            db.add(stage)
            db.flush()
            return _stage_to_dict(stage, [])
    except Exception as e:
        logger.error(f"Error creating stage: {e}")
        raise HTTPException(500, "Erreur lors de la création de l'étape.")


# ── PATCH /candidates/{candidate_id}/stage ────────────────────────────
@router.patch("/candidates/{candidate_id}/stage")
def move_candidate(candidate_id: str, body: StagePatch):
    """Déplace un candidat vers une autre étape du pipeline.
    Si le candidat existe uniquement dans le CSV (pas encore en DB),
    on crée une entrée minimale pour persister le stage_id."""
    try:
        with get_db() as db:
            candidate = db.get(CandidateModel, candidate_id)
            if candidate:
                candidate.stage_id = body.stage_id
            else:
                # Candidat en CSV seulement → on importe depuis le CSV si possible
                candidate = _import_from_csv(candidate_id, db)
                if candidate:
                    candidate.stage_id = body.stage_id
                else:
                    # Entrée minimale pour au moins stocker le stage
                    candidate = CandidateModel(
                        candidate_id=candidate_id,
                        stage_id=body.stage_id,
                    )
                    db.add(candidate)
            db.flush()
            return {"candidate_id": candidate_id, "stage_id": body.stage_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error moving candidate {candidate_id}: {e}")
        raise HTTPException(500, "Erreur lors du déplacement.")


def _import_from_csv(candidate_id: str, db) -> "CandidateModel | None":
    """Essaie d'importer un candidat depuis le CSV vers la DB."""
    try:
        import pandas as pd
        from api.config import CANDIDATES_FILE
        from datetime import datetime

        if not CANDIDATES_FILE.exists():
            return None
        df = pd.read_csv(CANDIDATES_FILE)
        row = df[df["candidate_id"] == candidate_id]
        if row.empty:
            return None
        r = row.iloc[0]

        def _f(col, default=None):
            v = r.get(col, default)
            return None if (v == "" or (isinstance(v, float) and pd.isna(v))) else v

        candidate = CandidateModel(
            candidate_id  = candidate_id,
            name          = _f("name"),
            email         = _f("email"),
            phone         = _f("phone"),
            gender        = _f("gender"),
            age           = int(_f("age")) if _f("age") is not None else None,
            sector        = _f("sector"),
            target_role   = _f("target_role"),
            years_experience = float(_f("years_experience")) if _f("years_experience") is not None else None,
            education_level  = float(_f("education_level"))  if _f("education_level")  is not None else None,
            score            = float(_f("score"))             if _f("score")             is not None else None,
            decision         = _f("decision"),
            threshold_used   = float(_f("threshold_used"))    if _f("threshold_used")    is not None else None,
            priority_rank    = int(_f("priority_rank"))        if _f("priority_rank")     is not None else None,
            eliminated_reason= _f("eliminated_reason"),
            status           = _f("status", "inbox"),
            source_filename  = _f("source_filename"),
            received_at      = datetime.utcnow(),
        )
        db.add(candidate)
        logger.info(f"Candidat {candidate_id} importé du CSV vers la DB.")
        return candidate
    except Exception as e:
        logger.warning(f"CSV import failed for {candidate_id}: {e}")
        return None
