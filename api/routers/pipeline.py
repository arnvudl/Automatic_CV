"""
routers/pipeline.py — Kanban pipeline : étapes par offre + déplacement candidats.

Endpoints :
  GET  /jobs/{job_id}/stages        → étapes du pipeline (crée les defaults si vide)
  POST /jobs/{job_id}/stages        → crée une étape custom
  PATCH /candidates/{id}/stage      → déplace un candidat vers une étape
"""

import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.auth import get_current_user
from api.database import get_db, PipelineStage as StageModel, Candidate as CandidateModel

router = APIRouter(tags=["pipeline"], dependencies=[Depends(get_current_user)])
logger = logging.getLogger("cv_api")

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

            # Uniquement les candidats explicitement assignés à ce job
            valid_stage_ids = {s.stage_id for s in stages}
            assigned = (
                db.query(CandidateModel)
                .filter(CandidateModel.stage_id.in_(valid_stage_ids))
                .all()
            )
            cand_briefs = [_candidate_brief(c) for c in assigned]

            return [_stage_to_dict(s, cand_briefs) for s in stages]

    except Exception as e:
        logger.error(f"Error getting stages for job {job_id}: {e}")
        raise HTTPException(500, "Erreur lors de la récupération des étapes.")


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
