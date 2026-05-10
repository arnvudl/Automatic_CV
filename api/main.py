"""
main.py — Point d'entrée FastAPI (app, CORS, SSE, scoring).
Routes non protégées : /, /events, /score, /api/v1/candidates (n8n)
Tout le reste → JWT requis (via dépendances router)
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.config import PROCESSED_DIR, RAW_TEXTS_DIR
from api.database import init_db, get_db, User as UserModel
from api.sse import _sse_clients, broadcast
from api.scoring import (
    model as _model,
    score_features,
    enrich_features,
    save_candidate,
)
from api.auth import hash_password
from api.routers import candidates, jobs, stats, comments
from api.routers import auth as auth_router
from api.routers import interviews as interviews_router

# Parsing pipeline
from pipeline_ml.core.p01_parse import parse_cv, parse_cv_llm, extract_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv_api")

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="CV-Intelligence API", version="2.1.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://ats.lony.app",
        "https://n8n.lony.app",
    ],
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────
app.include_router(auth_router.router)          # /auth/* — non protégé par défaut
app.include_router(candidates.router)           # /candidates/* — protégé via router deps
app.include_router(jobs.router)                 # /jobs/* — protégé
app.include_router(stats.router)                # /stats, /analyse/* — protégé
app.include_router(comments.router)             # /comments/* — protégé
app.include_router(interviews_router.router)    # /interviews/* — protégé


# ── Startup ──────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()
    _seed_admin()
    logger.info("Base de données initialisée.")


def _seed_admin():
    """Crée l'utilisateur admin depuis les variables d'env si aucun user n'existe."""
    admin_email    = os.getenv("ADMIN_EMAIL",    "admin@lony.app")
    admin_password = os.getenv("ADMIN_PASSWORD", "Luminary2025!")
    admin_name     = os.getenv("ADMIN_NAME",     "Admin RH")
    try:
        with get_db() as db:
            if db.query(UserModel).count() == 0:
                admin = UserModel(
                    user_id=uuid.uuid4().hex,
                    email=admin_email,
                    name=admin_name,
                    password_hash=hash_password(admin_password),
                    role="admin",
                )
                db.add(admin)
                logger.info(f"Admin créé : {admin_email}")
    except Exception as e:
        logger.warning(f"Seed admin failed: {e}")


# ── Status ───────────────────────────────────────────────────────────
@app.get("/", tags=["status"])
def root():
    return {
        "status":       "online",
        "model_loaded": _model is not None,
        "version":      "2.1.0",
        "endpoints":    ["/score", "/candidates", "/jobs", "/stats", "/interviews", "/auth/login", "/docs"],
    }


# ── SSE ───────────────────────────────────────────────────────────────
@app.get("/events", tags=["realtime"])
async def sse_stream():
    """Server-Sent Events — non protégé (EventSource n'envoie pas de headers custom)."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    _sse_clients.add(queue)

    async def generator():
        try:
            yield "event: connected\ndata: {}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=25)
                    import json
                    yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            _sse_clients.discard(queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── POST /score ───────────────────────────────────────────────────────
@app.post("/score", tags=["scoring"])
async def score_cv(file: UploadFile = File(...)):
    """Reçoit un CV (.txt, .pdf, .docx), retourne le score ML. Non protégé (n8n)."""
    if _model is None:
        raise HTTPException(503, "Modèle non disponible.")

    filename = file.filename or ""
    suffix   = Path(filename).suffix.lower()
    content  = await file.read()

    # Extraction texte selon le format — PDF/DOCX passent par extract_text()
    if suffix in (".pdf", ".docx", ".doc"):
        tmp_in = PROCESSED_DIR / f"_tmp_{uuid.uuid4().hex}{suffix}"
        try:
            tmp_in.write_bytes(content)
            text = extract_text(tmp_in)
        finally:
            tmp_in.unlink(missing_ok=True)
        if not text.strip():
            raise HTTPException(422, f"Impossible d'extraire le texte du fichier {filename} "
                                     "(PDF graphique sans couche texte — OCR requis).")
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

    try:
        id_row, feat_row = parse_cv_llm(text, filename=filename)
        logger.info(f"Parsing LLM OK — {id_row.get('name')}")
    except Exception as llm_err:
        logger.warning(f"Parsing LLM échoué ({llm_err}), fallback regex")
        tmp = PROCESSED_DIR / f"_tmp_{uuid.uuid4().hex}.txt"
        try:
            tmp.write_text(text, encoding="utf-8")
            id_row, feat_row = parse_cv(tmp, {})
        finally:
            tmp.unlink(missing_ok=True)

    age      = id_row.get("age")
    feat_row = enrich_features(feat_row, age)
    result   = score_features(feat_row, age)

    candidate_id = id_row["cv_id"]
    (RAW_TEXTS_DIR / f"{candidate_id}.txt").write_text(text, encoding="utf-8")

    record = {
        "candidate_id":    candidate_id,
        "received_at":     datetime.now().isoformat(),
        "source_filename": file.filename,
        **{k: id_row.get(k) for k in ["name", "email", "phone", "gender", "age"]},
        **{k: feat_row.get(k) for k in ["sector", "target_role", "years_experience", "education_level"]},
        **result,
        "status": "inbox",
    }
    save_candidate(record)
    logger.info(
        f"Scoré : {id_row.get('name')} → {result['decision']} "
        f"(score={result['score']:.3f}, rank={result['priority_rank']}e pct)"
    )

    response_payload = {
        "candidate_id":      candidate_id,
        "name":              id_row.get("name"),
        "email":             id_row.get("email"),
        "score":             result["score"],
        "priority_rank":     result["priority_rank"],
        "decision":          result["decision"],
        "eliminated_reason": result.get("eliminated_reason"),
        "threshold_used":    result["threshold_used"],
        "sector":            feat_row.get("sector"),
        "years_experience":  feat_row.get("years_experience"),
        "status":            "inbox",
        "received_at":       record["received_at"],
    }
    asyncio.create_task(broadcast({"type": "candidate_scored", "data": response_payload}))
    return response_payload


# ── POST /api/v1/candidates (legacy n8n) ─────────────────────────────
@app.post("/api/v1/candidates", tags=["n8n"])
async def receive_cv_n8n(file: UploadFile = File(...), filename: str = Form(...)):
    result = await score_cv(file)
    return {
        "status":       "scored",
        "candidate_id": result["candidate_id"],
        "filename":     filename,
        "name":         result.get("name"),
        "score":        result["score"],
        "decision":     result["decision"],
        "message":      f"Candidat scoré : {result['decision']} (score={result['score']:.3f})",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
