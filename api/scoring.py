"""
scoring.py — Chargement du modèle ML et tous les helpers de scoring.
Importé par main.py (/score) et par les routers d'analyse.
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from api.config import (
    MODELS_DIR, CANDIDATES_FILE, ELIMINATORY_FILE,
    CANDIDATE_FIELDS, RAW_TEXTS_DIR,
)

logger = logging.getLogger("cv_api")

# ── Groq client ──────────────────────────────────────────────────────
try:
    from groq import Groq as _Groq
    groq_client = _Groq(api_key=os.getenv("GROQ_API_KEY", ""))
except Exception:
    groq_client = None

GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Chargement modèle ML ─────────────────────────────────────────────
try:
    model     = joblib.load(MODELS_DIR / "model.pkl")
    scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
    features  = joblib.load(MODELS_DIR / "feature_cols.pkl")
    threshold = joblib.load(MODELS_DIR / "threshold.pkl")
    thr_jr    = joblib.load(MODELS_DIR / "threshold_junior.pkl") \
                if (MODELS_DIR / "threshold_junior.pkl").exists() else threshold
    logger.info("Modèle ML chargé.")
except Exception as e:
    logger.error(f"Impossible de charger le modèle : {e}")
    model = None

# ── Keywords secteur ─────────────────────────────────────────────────
IT_KW      = {"computer","software","data","information","it","computing",
               "engineering","technology","networks","ai"}
FINANCE_KW = {"finance","accounting","economics","business","management",
               "audit","banking"}


# ── Helpers scoring ──────────────────────────────────────────────────

def check_eliminatory(feat_row: dict) -> Optional[str]:
    """Vérifie les critères éliminatoires. Retourne la raison si éliminé, None sinon."""
    if not ELIMINATORY_FILE.exists():
        return None
    try:
        config = json.loads(ELIMINATORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    for rule in config.get("rules", []):
        if not rule.get("enabled"):
            continue
        field = rule["field"]
        op    = rule["operator"]
        val   = rule["value"]
        candidate_val = feat_row.get(field)
        if candidate_val is None:
            continue
        try:
            if op == "gte" and float(candidate_val) < float(val):
                return rule["reason"]
            if op == "in" and val and str(candidate_val) not in [str(v) for v in val]:
                return rule["reason"]
        except (TypeError, ValueError):
            pass
    return None


def priority_rank(score: float) -> int:
    """Retourne le rang percentile (0-100) du score parmi tous les candidats existants."""
    if not CANDIDATES_FILE.exists():
        return 50
    try:
        df = pd.read_csv(CANDIDATES_FILE)
        all_scores = pd.to_numeric(df["score"], errors="coerce").dropna().values
        if len(all_scores) == 0:
            return 50
        return int(round(np.sum(all_scores < score) / len(all_scores) * 100))
    except Exception:
        return 50


def score_features(feat: dict, age: Optional[int]) -> dict:
    """Applique le modèle ML sur un dict de features et retourne score + décision + SHAP."""
    if model is None:
        return {"score": None, "decision": "no_model", "threshold_used": None,
                "shap_json": "{}", "priority_rank": None, "eliminated_reason": None}

    # Critères éliminatoires — avant le ML
    elim = check_eliminatory(feat)
    if elim:
        return {"score": 0.0, "decision": "eliminated", "threshold_used": None,
                "shap_json": "{}", "priority_rank": 0, "eliminated_reason": elim}

    row = {k: float(feat.get(k, 0) or 0) for k in features}
    X   = np.array([[row[k] for k in features]])
    X_s = scaler.transform(X)
    score_val = float(model.predict_proba(X_s)[0][1])

    # SHAP individual
    shap_dict = {}
    try:
        import shap as _shap
        explainer = _shap.LinearExplainer(model, X_s, feature_perturbation="interventional")
        sv = explainer.shap_values(X_s)
        raw = sv[0] if isinstance(sv, list) else sv[0]
        shap_dict = {k: round(float(v), 4) for k, v in zip(features, raw)}
    except Exception:
        pass

    is_junior = (age is not None and age < 30)
    thr = thr_jr if is_junior else threshold
    decision = "invite" if score_val >= thr else "reject"

    return {
        "score":             round(score_val, 4),
        "decision":          decision,
        "threshold_used":    round(thr, 4),
        "shap_json":         json.dumps(shap_dict),
        "priority_rank":     priority_rank(score_val),
        "eliminated_reason": None,
    }


def enrich_features(feat_row: dict, age: Optional[int] = None) -> dict:
    """Feature engineering v4 pour l'inférence live (miroir de p02_features)."""
    years   = float(feat_row.get("years_experience") or 0)
    avg_dur = float(feat_row.get("avg_job_duration") or 0)
    nb_jobs = max(float(feat_row.get("nb_jobs") or 1), 1)
    nb_tech = float(feat_row.get("nb_technical_skills") or 0)
    nb_meth = float(feat_row.get("nb_methods_skills") or 0)
    nb_cert = float(feat_row.get("nb_certifications") or 0)
    nb_lang = float(feat_row.get("nb_languages") or 0)
    eng_lvl = float(feat_row.get("english_level") or 0)
    sector  = feat_row.get("sector", "Other")
    edu_field = str(feat_row.get("education_field") or "").lower()

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
    career_years = max((age or 30) - 22, 1)
    feat_row["exp_per_year_of_age"]    = round(years / career_years, 3)
    if sector == "IT":
        feat_row["field_match"] = int(any(k in edu_field for k in IT_KW))
    elif sector == "Finance":
        feat_row["field_match"] = int(any(k in edu_field for k in FINANCE_KW))
    else:
        feat_row["field_match"] = 0
    edu_raw = float(feat_row.get("education_level") or 2)
    _edu_map = {1: 0.0, 2: 0.30, 3: 0.70, 4: 0.80}
    feat_row["education_adj"] = _edu_map.get(round(edu_raw), 0.30)

    return feat_row


def save_candidate(record: dict):
    """
    Persiste un candidat :
    1. PostgreSQL / SQLite via SQLAlchemy (source de vérité)
    2. CSV legacy en parallèle (rétrocompatibilité pipeline ML)
    """
    from api.database import get_db, Candidate as CandidateModel

    # ── DB ────────────────────────────────────────────────
    try:
        with get_db() as db:
            existing = db.get(CandidateModel, record["candidate_id"])
            if existing:
                for k, v in record.items():
                    if hasattr(existing, k):
                        setattr(existing, k, v)
            else:
                received = record.get("received_at")
                if isinstance(received, str):
                    try:
                        received = datetime.fromisoformat(received)
                    except ValueError:
                        received = datetime.utcnow()
                obj = CandidateModel(
                    candidate_id=record["candidate_id"],
                    received_at=received,
                    source_filename=record.get("source_filename"),
                    name=record.get("name"),
                    email=record.get("email"),
                    phone=record.get("phone"),
                    gender=record.get("gender"),
                    age=record.get("age"),
                    sector=record.get("sector"),
                    target_role=record.get("target_role"),
                    years_experience=record.get("years_experience"),
                    education_level=record.get("education_level"),
                    score=record.get("score"),
                    decision=record.get("decision"),
                    threshold_used=record.get("threshold_used"),
                    priority_rank=record.get("priority_rank"),
                    eliminated_reason=record.get("eliminated_reason"),
                    shap_json=record.get("shap_json"),
                    status=record.get("status", "inbox"),
                )
                db.add(obj)
    except Exception as db_err:
        logger.warning(f"DB write failed ({db_err}) — CSV fallback only")

    # ── CSV fallback ──────────────────────────────────────
    exists = CANDIDATES_FILE.exists()
    try:
        with CANDIDATES_FILE.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(record)
    except Exception as csv_err:
        logger.warning(f"CSV write failed: {csv_err}")
