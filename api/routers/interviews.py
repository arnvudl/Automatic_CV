"""
routers/interviews.py — CRUD entretiens planifiés.
"""

import uuid
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from api.auth import get_current_user
from api.database import get_db, Interview as InterviewModel

router = APIRouter(tags=["interviews"], dependencies=[Depends(get_current_user)])
logger = logging.getLogger("cv_api")


def _to_dict(i: InterviewModel) -> dict:
    return {
        "interview_id":   i.interview_id,
        "candidate_id":   i.candidate_id,
        "candidate_name": i.candidate_name,
        "date":           i.date,
        "time":           i.time,
        "type":           i.interview_type,
        "notes":          i.notes,
        "created_at":     i.created_at.isoformat() if i.created_at else None,
        "updated_at":     i.updated_at.isoformat() if i.updated_at else None,
    }


class InterviewCreate(BaseModel):
    candidate_id:   str
    candidate_name: Optional[str] = None
    date:           str   # YYYY-MM-DD
    time:           Optional[str] = None  # HH:MM
    type:           Optional[str] = "Entretien technique"
    notes:          Optional[str] = None

class InterviewUpdate(BaseModel):
    date:  Optional[str] = None
    time:  Optional[str] = None
    type:  Optional[str] = None
    notes: Optional[str] = None


@router.get("/interviews")
def list_interviews(
    candidate_id: Optional[str] = Query(None),
    month:        Optional[str] = Query(None, description="YYYY-MM"),
):
    """Liste tous les entretiens planifiés, optionnellement filtrés."""
    try:
        with get_db() as db:
            q = db.query(InterviewModel)
            if candidate_id:
                q = q.filter(InterviewModel.candidate_id == candidate_id)
            if month:
                q = q.filter(InterviewModel.date.startswith(month))
            interviews = q.order_by(InterviewModel.date, InterviewModel.time).all()
            return [_to_dict(i) for i in interviews]
    except Exception as e:
        logger.error(f"Error listing interviews: {e}")
        raise HTTPException(500, "Erreur lors de la récupération des entretiens.")


@router.post("/interviews", status_code=201)
def create_interview(body: InterviewCreate):
    """Crée un entretien planifié."""
    try:
        with get_db() as db:
            interview = InterviewModel(
                interview_id=uuid.uuid4().hex,
                candidate_id=body.candidate_id,
                candidate_name=body.candidate_name,
                date=body.date,
                time=body.time,
                interview_type=body.type,
                notes=body.notes,
                created_at=datetime.utcnow(),
            )
            db.add(interview)
            db.flush()
            return _to_dict(interview)
    except Exception as e:
        logger.error(f"Error creating interview: {e}")
        raise HTTPException(500, "Erreur lors de la création de l'entretien.")


@router.patch("/interviews/{interview_id}")
def update_interview(interview_id: str, body: InterviewUpdate):
    """Modifie un entretien (date, heure, type, notes)."""
    try:
        with get_db() as db:
            interview = db.get(InterviewModel, interview_id)
            if not interview:
                raise HTTPException(404, f"Entretien {interview_id} introuvable.")
            if body.date  is not None: interview.date           = body.date
            if body.time  is not None: interview.time           = body.time
            if body.type  is not None: interview.interview_type = body.type
            if body.notes is not None: interview.notes          = body.notes
            interview.updated_at = datetime.utcnow()
            db.flush()
            return _to_dict(interview)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating interview: {e}")
        raise HTTPException(500, "Erreur lors de la mise à jour.")


@router.delete("/interviews/{interview_id}")
def delete_interview(interview_id: str):
    """Supprime / annule un entretien."""
    try:
        with get_db() as db:
            interview = db.get(InterviewModel, interview_id)
            if not interview:
                raise HTTPException(404, f"Entretien {interview_id} introuvable.")
            db.delete(interview)
        return {"deleted": True, "interview_id": interview_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting interview: {e}")
        raise HTTPException(500, "Erreur lors de la suppression.")
