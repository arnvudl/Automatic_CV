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
from api.routers import pipeline as pipeline_router
from api.routers import scorecards as scorecards_router

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
app.include_router(pipeline_router.router)      # /jobs/{id}/stages, /candidates/{id}/stage — protégé
app.include_router(scorecards_router.router)    # /candidates/{id}/scorecards, /scorecards/{id} — protégé


# ── Startup ──────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()
    _seed_admin()
    _migrate_csv_to_db()
    _recompute_shap()
    logger.info("Base de données initialisée.")


def _recompute_shap():
    """Recalcule le SHAP pour les candidats en DB qui ont shap_json vide ou '{}'."""
    from api.database import get_db, Candidate as CandidateModel
    from api.scoring import score_features, enrich_features

    try:
        with get_db() as db:
            candidates = db.query(CandidateModel).filter(
                (CandidateModel.shap_json == None) |
                (CandidateModel.shap_json == "{}") |
                (CandidateModel.shap_json == "")
            ).all()

            if not candidates:
                return

            updated = 0
            for c in candidates:
                if c.score is None:
                    continue
                try:
                    feat_row = {
                        "years_experience": float(c.years_experience or 0),
                        "education_level":  float(c.education_level  or 0),
                        "sector":           c.sector or "",
                        "target_role":      c.target_role or "",
                        "avg_job_duration": 0,   # non stocké, approximation
                        "career_depth":     0,
                    }
                    feat_row   = enrich_features(feat_row, c.age)
                    result     = score_features(feat_row, c.age)
                    shap_json  = result.get("shap_json", "{}")
                    if shap_json and shap_json != "{}":
                        c.shap_json = shap_json
                        updated += 1
                except Exception:
                    pass

            if updated:
                logger.info(f"SHAP recomputed pour {updated} candidat(s).")
    except Exception as e:
        logger.warning(f"SHAP recomputation failed: {e}")


def _migrate_csv_to_db():
    """Migration unique : importe les candidats du CSV vers la DB s'ils n'y sont pas déjà."""
    from api.config import CANDIDATES_FILE
    from api.database import get_db, Candidate as CandidateModel
    from api.scoring import save_candidate
    import pandas as pd

    if not CANDIDATES_FILE.exists():
        return
    try:
        df = pd.read_csv(CANDIDATES_FILE)
        if df.empty:
            return
        with get_db() as db:
            existing_ids = {r.candidate_id for r in db.query(CandidateModel.candidate_id).all()}

        to_import = df[~df["candidate_id"].isin(existing_ids)]
        if to_import.empty:
            return

        for _, row in to_import.iterrows():
            try:
                save_candidate(row.dropna().to_dict())
            except Exception as e:
                logger.warning(f"Skipping CSV candidate {row.get('candidate_id')}: {e}")

        logger.info(f"{len(to_import)} candidat(s) migrés du CSV vers la DB.")
    except Exception as e:
        logger.warning(f"Migration CSV→DB échouée : {e}")


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

    import json as _json

    cv_extra = {}
    try:
        id_row, feat_row = parse_cv_llm(text, filename=filename)
        logger.info(f"Parsing LLM OK — {id_row.get('name')}")
        cv_extra = id_row.pop("_cv_extra", {}) or {}
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

    # Si le LLM n'a pas fourni les données enrichies, on les parse depuis le texte brut
    if not cv_extra:
        from api.routers.candidates import _parse_cv_detail
        try:
            cv_extra = _parse_cv_detail(file.filename or "", candidate_id) or {}
        except Exception:
            cv_extra = {}

    record = {
        "candidate_id":    candidate_id,
        "received_at":     datetime.now().isoformat(),
        "source_filename": file.filename,
        **{k: id_row.get(k) for k in ["name", "email", "phone", "gender", "age"]},
        **{k: feat_row.get(k) for k in ["sector", "target_role", "years_experience", "education_level"]},
        **result,
        "status":        "inbox",
        "cv_extra_json": _json.dumps(cv_extra, ensure_ascii=False) if cv_extra else None,
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
