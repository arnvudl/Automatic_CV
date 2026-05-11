"""
routers/scorecards.py — Fiches d'évaluation RH par candidat.

Endpoints :
  GET    /candidates/{id}/scorecards        → liste des évaluations
  POST   /candidates/{id}/scorecards        → créer une évaluation
  DELETE /scorecards/{scorecard_id}         → supprimer une évaluation
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.auth import get_current_user
from api.database import get_db, Scorecard as ScorecardModel

router = APIRouter(tags=["scorecards"], dependencies=[Depends(get_current_user)])
logger = logging.getLogger("cv_api")

CRITERIA = [
    {"key": "technique",     "label": "Compétences techniques"},
    {"key": "communication", "label": "Communication"},
    {"key": "motivation",    "label": "Motivation"},
    {"key": "soft_skills",   "label": "Soft skills"},
    {"key": "adequation",    "label": "Adéquation au poste"},
]


def _to_dict(s: ScorecardModel) -> dict:
    ratings = {}
    try:
        ratings = json.loads(s.ratings) if s.ratings else {}
    except Exception:
        pass
    return {
        "scorecard_id":   s.scorecard_id,
        "candidate_id":   s.candidate_id,
        "evaluator_name": s.evaluator_name,
        "ratings":        ratings,
        "notes":          s.notes,
        "overall":        s.overall,
        "created_at":     s.created_at.isoformat() if s.created_at else None,
    }


class ScorecardCreate(BaseModel):
    evaluator_name: Optional[str] = "RH"
    ratings:        dict          = {}   # {key: 1-5}
    notes:          Optional[str] = None


# ── GET /candidates/{id}/scorecards ──────────────────────────────────
@router.get("/candidates/{candidate_id}/scorecards")
def list_scorecards(candidate_id: str):
    try:
        with get_db() as db:
            rows = (
                db.query(ScorecardModel)
                .filter(ScorecardModel.candidate_id == candidate_id)
                .order_by(ScorecardModel.created_at.desc())
                .all()
            )
            return {"criteria": CRITERIA, "scorecards": [_to_dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"Error listing scorecards: {e}")
        raise HTTPException(500, "Erreur lors de la récupération des évaluations.")


# ── POST /candidates/{id}/scorecards ─────────────────────────────────
@router.post("/candidates/{candidate_id}/scorecards", status_code=201)
def create_scorecard(candidate_id: str, body: ScorecardCreate):
    try:
        # Calcul de la moyenne sur les critères connus
        valid = {c["key"] for c in CRITERIA}
        scores = [v for k, v in body.ratings.items() if k in valid and isinstance(v, (int, float)) and 1 <= v <= 5]
        overall = round(sum(scores) / len(scores), 2) if scores else None

        with get_db() as db:
            sc = ScorecardModel(
                candidate_id   = candidate_id,
                evaluator_name = body.evaluator_name or "RH",
                ratings        = json.dumps(body.ratings),
                notes          = body.notes,
                overall        = overall,
            )
            db.add(sc)
            db.flush()
            return _to_dict(sc)
    except Exception as e:
        logger.error(f"Error creating scorecard: {e}")
        raise HTTPException(500, "Erreur lors de la création de l'évaluation.")


# ── DELETE /scorecards/{scorecard_id} ────────────────────────────────
@router.delete("/scorecards/{scorecard_id}")
def delete_scorecard(scorecard_id: str):
    try:
        with get_db() as db:
            sc = db.get(ScorecardModel, scorecard_id)
            if not sc:
                raise HTTPException(404, "Évaluation introuvable.")
            db.delete(sc)
        return {"deleted": True, "scorecard_id": scorecard_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting scorecard: {e}")
        raise HTTPException(500, "Erreur lors de la suppression.")
