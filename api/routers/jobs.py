"""
routers/jobs.py — CRUD offres d'emploi.
"""

import uuid
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.database import get_db, Job as JobModel

router = APIRouter(tags=["jobs"])
logger = logging.getLogger("cv_api")

# ── Helpers ──────────────────────────────────────────────────────────
STAGE_PROGRESS = {
    "sourcing":        15,
    "review":          40,
    "interview":       65,
    "final_interview": 85,
    "closed":         100,
}


def _job_to_dict(j: JobModel) -> dict:
    return {
        "job_id":           j.job_id,
        "title":            j.title,
        "department":       j.department,
        "location":         j.location,
        "description":      j.description,
        "status":           j.status,
        "stage":            j.stage,
        "priority":         j.priority,
        "applicants_count": j.applicants_count or 0,
        "avg_score":        j.avg_score,
        "progress":         STAGE_PROGRESS.get(j.stage or "sourcing", 15),
        "created_at":       j.created_at.isoformat() if j.created_at else None,
        "updated_at":       j.updated_at.isoformat() if j.updated_at else None,
    }


# ── Pydantic ──────────────────────────────────────────────────────────
class JobCreate(BaseModel):
    title:            str
    department:       Optional[str]   = None
    location:         Optional[str]   = None
    description:      Optional[str]   = None
    status:           str             = "active"
    stage:            str             = "sourcing"
    priority:         str             = "normal"
    applicants_count: int             = 0
    avg_score:        Optional[float] = None


class JobUpdate(BaseModel):
    title:            Optional[str]   = None
    department:       Optional[str]   = None
    location:         Optional[str]   = None
    description:      Optional[str]   = None
    status:           Optional[str]   = None
    stage:            Optional[str]   = None
    priority:         Optional[str]   = None
    applicants_count: Optional[int]   = None
    avg_score:        Optional[float] = None


# ── Endpoints ─────────────────────────────────────────────────────────
@router.get("/jobs")
def list_jobs(status: Optional[str] = Query(None)):
    try:
        with get_db() as db:
            q = db.query(JobModel)
            if status:
                q = q.filter(JobModel.status == status)
            jobs = q.order_by(JobModel.created_at.desc()).all()
            return [_job_to_dict(j) for j in jobs]
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(500, "Erreur lors de la récupération des offres.")


@router.post("/jobs", status_code=201)
def create_job(body: JobCreate):
    try:
        with get_db() as db:
            job = JobModel(
                job_id=uuid.uuid4().hex,
                title=body.title,
                department=body.department,
                location=body.location,
                description=body.description,
                status=body.status,
                stage=body.stage,
                priority=body.priority,
                applicants_count=body.applicants_count,
                avg_score=body.avg_score,
                created_at=datetime.utcnow(),
            )
            db.add(job)
            db.flush()
            return _job_to_dict(job)
    except Exception as e:
        logger.error(f"Error creating job: {e}")
        raise HTTPException(500, "Erreur lors de la création de l'offre.")


@router.patch("/jobs/{job_id}")
def update_job(job_id: str, body: JobUpdate):
    try:
        with get_db() as db:
            job = db.get(JobModel, job_id)
            if not job:
                raise HTTPException(404, f"Offre {job_id} introuvable.")
            for field, val in body.model_dump(exclude_unset=True).items():
                setattr(job, field, val)
            job.updated_at = datetime.utcnow()
            db.flush()
            return _job_to_dict(job)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating job: {e}")
        raise HTTPException(500, "Erreur lors de la mise à jour.")


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    try:
        with get_db() as db:
            job = db.get(JobModel, job_id)
            if not job:
                raise HTTPException(404, f"Offre {job_id} introuvable.")
            db.delete(job)
        return {"deleted": True, "job_id": job_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job: {e}")
        raise HTTPException(500, "Erreur lors de la suppression.")
