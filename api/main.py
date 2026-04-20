"""
API CV-Intelligence — FastAPI
Reçoit un CV texte, le score avec le modèle ML, retourne la décision.
Endpoints consommés par N8N et la page RH.
"""

import csv
import json
import uuid
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Imports pipeline ────────────────────────────────────────────
import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from pipeline_ml.core.p01_parse import parse_cv

# ── Config ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv_api")

MODELS_DIR      = ROOT / "models"
PROCESSED_DIR   = ROOT / "data" / "processed"
CANDIDATES_FILE = PROCESSED_DIR / "candidates_live.csv"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Chargement modèle ────────────────────────────────────────────
try:
    _model     = joblib.load(MODELS_DIR / "model.pkl")
    _scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
    _features  = joblib.load(MODELS_DIR / "feature_cols.pkl")
    _threshold = joblib.load(MODELS_DIR / "threshold.pkl")
    _thr_jr    = joblib.load(MODELS_DIR / "threshold_junior.pkl") if (MODELS_DIR / "threshold_junior.pkl").exists() else _threshold
    logger.info("Modèle ML chargé.")
except Exception as e:
    logger.error(f"Impossible de charger le modèle : {e}")
    _model = None

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(title="CV-Intelligence API", version="1.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restreindre en prod : ["https://n8n.lony.app"]
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────

def _score_features(feat: dict, age: Optional[int]) -> dict:
    """Applique le modèle ML sur un dict de features et retourne score + décision."""
    if _model is None:
        return {"score": None, "decision": "no_model", "threshold_used": None}

    row = {k: float(feat.get(k, 0) or 0) for k in _features}
    X = np.array([[row[k] for k in _features]])
    X_s = _scaler.transform(X)
    score = float(_model.predict_proba(X_s)[0][1])

    is_junior = (age is not None and age < 30)
    thr = _thr_jr if is_junior else _threshold
    decision = "invite" if score >= thr else "reject"
    return {"score": round(score, 4), "decision": decision, "threshold_used": round(thr, 4)}


def _save_candidate(record: dict):
    """Ajoute un candidat au fichier CSV live."""
    fieldnames = [
        "candidate_id", "received_at", "source_filename", "name", "email", "phone",
        "gender", "age", "sector", "target_role", "years_experience", "education_level",
        "score", "decision", "threshold_used",
    ]
    exists = CANDIDATES_FILE.exists()
    with CANDIDATES_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def _enrich_features(feat_row: dict) -> dict:
    """Feature engineering minimal pour l'inférence live (miroir de p02_features)."""
    years   = float(feat_row.get("years_experience") or 0)
    avg_dur = float(feat_row.get("avg_job_duration") or 0)
    nb_jobs = max(float(feat_row.get("nb_jobs") or 1), 1)
    nb_tech = float(feat_row.get("nb_technical_skills") or 0)
    nb_meth = float(feat_row.get("nb_methods_skills") or 0)
    nb_cert = float(feat_row.get("nb_certifications") or 0)
    nb_lang = float(feat_row.get("nb_languages") or 0)
    eng_lvl = float(feat_row.get("english_level") or 0)
    sector  = feat_row.get("sector", "Other")

    feat_row["log_years_exp"]          = round(np.log1p(years), 3)
    feat_row["log_avg_job_duration"]   = round(np.log1p(avg_dur), 3)
    feat_row["has_multiple_languages"] = int(nb_lang >= 2)
    potential = round((nb_tech + nb_meth + nb_cert) / (years + 1), 3)
    feat_row["potential_score"]        = potential
    feat_row["junior_potential"]       = round(int(years < 3) * potential, 3)
    feat_row["cert_density"]           = round(nb_cert / nb_jobs, 3)
    feat_row["multilingual_score"]     = int(nb_lang + int(eng_lvl >= 4))
    feat_row["method_tech_ratio"]      = round(nb_meth / max(nb_tech, 1), 3)
    feat_row["tech_per_year"]          = round(nb_tech / max(years, 0.5), 3)
    feat_row["career_depth"]           = round(years * avg_dur, 3)
    feat_row["is_it"]                  = int(sector == "IT")
    feat_row["is_finance"]             = int(sector == "Finance")
    return feat_row


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/", tags=["status"])
def root():
    return {
        "status": "online",
        "model_loaded": _model is not None,
        "endpoints": ["/score", "/api/v1/candidates", "/candidates", "/docs"],
    }


@app.post("/score", tags=["scoring"])
async def score_cv(file: UploadFile = File(...)):
    """
    Reçoit un fichier .txt (CV), retourne le score ML et la décision.
    Utilisé par N8N directement.
    """
    if _model is None:
        raise HTTPException(503, "Modèle non disponible — relancez le pipeline.")

    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    # Sauvegarde temporaire pour parse_cv (qui lit un fichier)
    tmp = PROCESSED_DIR / f"_tmp_{uuid.uuid4().hex}.txt"
    try:
        tmp.write_text(text, encoding="utf-8")
        id_row, feat_row = parse_cv(tmp, {})
    finally:
        tmp.unlink(missing_ok=True)

    feat_row = _enrich_features(feat_row)
    age = id_row.get("age")
    result = _score_features(feat_row, age)

    record = {
        "candidate_id": id_row["cv_id"],
        "received_at":  datetime.now().isoformat(),
        "source_filename": file.filename,
        **{k: id_row.get(k) for k in ["name", "email", "phone", "gender", "age"]},
        **{k: feat_row.get(k) for k in ["sector", "target_role", "years_experience", "education_level"]},
        **result,
    }
    _save_candidate(record)
    logger.info(f"Scoré : {id_row.get('name')} → {result['decision']} ({result['score']:.3f})")

    return {
        "candidate_id": id_row["cv_id"],
        "name":         id_row.get("name"),
        "email":        id_row.get("email"),
        "score":        result["score"],
        "decision":     result["decision"],
        "threshold_used": result["threshold_used"],
        "sector":       feat_row.get("sector"),
        "years_experience": feat_row.get("years_experience"),
    }


@app.post("/api/v1/candidates", tags=["n8n"])
async def receive_cv_n8n(
    file: UploadFile = File(...),
    filename: str = Form(...),
    **kwargs,
):
    """
    Endpoint legacy compatible avec le workflow N8N existant.
    Appelle /score en interne et retourne la même structure qu'avant + score ML.
    """
    result = await score_cv(file)
    return {
        "status": "scored",
        "candidate_id": result["candidate_id"],
        "filename": filename,
        "name": result.get("name"),
        "score": result["score"],
        "decision": result["decision"],
        "message": f"Candidat scoré : {result['decision']} (score={result['score']:.3f})",
    }


@app.get("/candidates", tags=["rh"])
def list_candidates(
    decision: Optional[str] = Query(None, description="invite | reject"),
    sector:   Optional[str] = Query(None),
    min_score: float = Query(0.0),
):
    """Retourne la liste des candidats scorés (pour la page RH)."""
    if not CANDIDATES_FILE.exists():
        return []
    df = pd.read_csv(CANDIDATES_FILE)
    if decision:
        df = df[df["decision"] == decision]
    if sector:
        df = df[df["sector"].str.lower() == sector.lower()]
    df = df[pd.to_numeric(df["score"], errors="coerce").fillna(0) >= min_score]
    df = df.sort_values("score", ascending=False)
    return df.fillna("").to_dict(orient="records")


@app.get("/stats", tags=["rh"])
def get_stats():
    """Stats globales pour le dashboard RH."""
    if not CANDIDATES_FILE.exists():
        return {"total": 0, "invited": 0, "rejected": 0, "invite_rate": 0}
    df = pd.read_csv(CANDIDATES_FILE)
    total    = len(df)
    invited  = int((df["decision"] == "invite").sum())
    rejected = int((df["decision"] == "reject").sum())
    return {
        "total":       total,
        "invited":     invited,
        "rejected":    rejected,
        "invite_rate": round(invited / total * 100, 1) if total else 0,
        "by_sector":   df.groupby("sector")["decision"].apply(
            lambda x: {"total": len(x), "invited": int((x == "invite").sum())}
        ).to_dict(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
