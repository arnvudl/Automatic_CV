"""
API CV-Intelligence — FastAPI
Reçoit un CV texte, le score avec le modèle ML, retourne la décision.
Endpoints consommés par N8N et la page RH.
"""

import csv
import json
import os
import uuid
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List

# Charger .env depuis la racine du projet (deux niveaux au-dessus de api/)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv non installé — variables d'env déjà définies dans le shell

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    import anthropic as _anthropic
    _anthropic_client = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
except Exception:
    _anthropic_client = None

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
RAW_TEXTS_DIR   = PROCESSED_DIR / "raw_texts"   # CV texte brut par candidate_id
CANDIDATES_FILE = PROCESSED_DIR / "candidates_live.csv"
COMMENTS_FILE   = PROCESSED_DIR / "comments.json"
CONFIG_DIR      = ROOT / "config"
ELIMINATORY_FILE = CONFIG_DIR / "eliminatory_criteria.json"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_TEXTS_DIR.mkdir(parents=True, exist_ok=True)

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
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────

def _check_eliminatory(feat_row: dict) -> Optional[str]:
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


def _priority_rank(score: float) -> int:
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


def _score_features(feat: dict, age: Optional[int]) -> dict:
    """Applique le modèle ML sur un dict de features et retourne score + décision + SHAP."""
    if _model is None:
        return {"score": None, "decision": "no_model", "threshold_used": None,
                "shap_json": "{}", "priority_rank": None, "eliminated_reason": None}

    # Critères éliminatoires — avant le ML
    elim = _check_eliminatory(feat)
    if elim:
        return {"score": 0.0, "decision": "eliminated", "threshold_used": None,
                "shap_json": "{}", "priority_rank": 0, "eliminated_reason": elim}

    row = {k: float(feat.get(k, 0) or 0) for k in _features}
    X = np.array([[row[k] for k in _features]])
    X_s = _scaler.transform(X)
    score = float(_model.predict_proba(X_s)[0][1])

    # SHAP individual
    shap_dict = {}
    try:
        import shap as _shap
        explainer = _shap.LinearExplainer(_model, X_s, feature_perturbation="interventional")
        sv = explainer.shap_values(X_s)
        raw = sv[0] if isinstance(sv, list) else sv[0]
        shap_dict = {k: round(float(v), 4) for k, v in zip(_features, raw)}
    except Exception:
        pass

    is_junior = (age is not None and age < 30)
    thr = _thr_jr if is_junior else _threshold
    decision = "invite" if score >= thr else "reject"

    return {
        "score":            round(score, 4),
        "decision":         decision,
        "threshold_used":   round(thr, 4),
        "shap_json":        json.dumps(shap_dict),
        "priority_rank":    _priority_rank(score),
        "eliminated_reason": None,
    }


CANDIDATE_FIELDS = [
    "candidate_id", "received_at", "source_filename", "name", "email", "phone",
    "gender", "age", "sector", "target_role", "years_experience", "education_level",
    "score", "decision", "threshold_used", "status", "shap_json",
    "priority_rank", "eliminated_reason",
]

def _save_candidate(record: dict):
    """Ajoute un candidat au fichier CSV live."""
    fieldnames = CANDIDATE_FIELDS
    exists = CANDIDATES_FILE.exists()
    with CANDIDATES_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(record)


IT_KW      = {"computer","software","data","information","it","computing","engineering","technology","networks","ai"}
FINANCE_KW = {"finance","accounting","economics","business","management","audit","banking"}

def _enrich_features(feat_row: dict, age: Optional[int] = None) -> dict:
    """Feature engineering v3 pour l'inférence live (miroir de p02_features)."""
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
    # v3 features
    career_years = max((age or 30) - 22, 1)
    feat_row["exp_per_year_of_age"]    = round(years / career_years, 3)
    if sector == "IT":
        feat_row["field_match"] = int(any(k in edu_field for k in IT_KW))
    elif sector == "Finance":
        feat_row["field_match"] = int(any(k in edu_field for k in FINANCE_KW))
    else:
        feat_row["field_match"] = 0

    # v4 features — education_adj (compressed scale, RRK-inspired)
    edu_raw = float(feat_row.get("education_level") or 2)
    _edu_map = {1: 0.0, 2: 0.30, 3: 0.70, 4: 0.80}
    feat_row["education_adj"] = _edu_map.get(round(edu_raw), 0.30)

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

    age = id_row.get("age")
    feat_row = _enrich_features(feat_row, age)
    result = _score_features(feat_row, age)

    candidate_id = id_row["cv_id"]

    # Sauvegarde du texte brut pour analyse LLM future
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
    _save_candidate(record)
    logger.info(f"Scoré : {id_row.get('name')} → {result['decision']} "
                f"(score={result['score']:.3f}, rank={result['priority_rank']}e percentile)")

    return {
        "candidate_id":    candidate_id,
        "name":            id_row.get("name"),
        "email":           id_row.get("email"),
        "score":           result["score"],
        "priority_rank":   result["priority_rank"],
        "decision":        result["decision"],
        "eliminated_reason": result.get("eliminated_reason"),
        "threshold_used":  result["threshold_used"],
        "sector":          feat_row.get("sector"),
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


class StatusUpdate(BaseModel):
    status: str  # inbox | review | interview | rejected

@app.patch("/candidates/{candidate_id}/status", tags=["rh"])
def update_candidate_status(candidate_id: str, body: StatusUpdate):
    """Met à jour le statut Kanban d'un candidat."""
    valid = {"inbox", "review", "interview", "rejected"}
    if body.status not in valid:
        raise HTTPException(400, f"Statut invalide. Valeurs acceptées : {valid}")
    if not CANDIDATES_FILE.exists():
        raise HTTPException(404, "Aucun candidat enregistré.")
    df = pd.read_csv(CANDIDATES_FILE)
    if candidate_id not in df["candidate_id"].values:
        raise HTTPException(404, f"Candidat {candidate_id} introuvable.")
    if "status" not in df.columns:
        df["status"] = "inbox"
    df.loc[df["candidate_id"] == candidate_id, "status"] = body.status
    df.to_csv(CANDIDATES_FILE, index=False)
    return {"candidate_id": candidate_id, "status": body.status}

@app.get("/stats", tags=["rh"])
def get_stats():
    """Stats globales pour le dashboard RH."""
    if not CANDIDATES_FILE.exists():
        return {"total": 0, "invited": 0, "rejected": 0, "invite_rate": 0,
                "today": 0, "borderline": 0, "pending_review": 0, "avg_score": 0}
    df = pd.read_csv(CANDIDATES_FILE)
    if "status" not in df.columns:
        df["status"] = "inbox"
    df["score_num"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    df["thr_num"]   = pd.to_numeric(df["threshold_used"], errors="coerce").fillna(0.5)

    today    = datetime.now().date().isoformat()
    total    = len(df)
    invited  = int((df["decision"] == "invite").sum())
    rejected = int((df["decision"] == "reject").sum())

    # Borderline : score dans les 8% du seuil
    df["gap"] = (df["score_num"] - df["thr_num"]).abs()
    borderline = int((df["gap"] <= 0.08).sum())

    # En attente de revue humaine : statut inbox ou review
    pending = int(df["status"].isin(["inbox", "review"]).sum())

    # Aujourd'hui
    today_mask = df["received_at"].astype(str).str.startswith(today)
    today_count = int(today_mask.sum())

    # Temps moyen de traitement (fictif ici — en prod: delta parse → score)
    avg_score = round(float(df["score_num"].mean()), 3) if total else 0

    # Alertes : rejetés avec score > 0.3 (profils potentiellement intéressants)
    interesting_rejected = df[
        (df["decision"] == "reject") & (df["score_num"] >= 0.3)
    ][["candidate_id", "name", "score", "target_role", "sector"]].head(5).fillna("").to_dict(orient="records")

    # Borderline list
    borderline_list = df[df["gap"] <= 0.08][
        ["candidate_id", "name", "score", "threshold_used", "target_role", "decision"]
    ].head(5).fillna("").to_dict(orient="records")

    return {
        "total":               total,
        "invited":             invited,
        "rejected":            rejected,
        "invite_rate":         round(invited / total * 100, 1) if total else 0,
        "today":               today_count,
        "borderline":          borderline,
        "pending_review":      pending,
        "avg_score":           avg_score,
        "interesting_rejected": interesting_rejected,
        "borderline_list":     borderline_list,
        "by_sector": {
            str(sector_name): {"total": len(grp), "invited": int((grp["decision"] == "invite").sum())}
            for sector_name, grp in df.groupby("sector", dropna=True)
        },
    }


RAW_DIR = ROOT / "data" / "raw"

import re as _re
_RE_SUMMARY  = _re.compile(r"Professional Summary:\s*\n(.*?)(?:\n\n|\nEducation:|\nExperience:)", _re.DOTALL)
_RE_TECH     = _re.compile(r"Technical:\s*(.*)", _re.IGNORECASE)
_RE_METH     = _re.compile(r"Methods:\s*(.*)",   _re.IGNORECASE)
_RE_MAN      = _re.compile(r"Management:\s*(.*)", _re.IGNORECASE)
_RE_LANG_BLK = _re.compile(r"Languages:(.*?)(?:Certifications:|$)", _re.DOTALL | _re.IGNORECASE)
_RE_CERT_BLK = _re.compile(r"Certifications:(.*?)$", _re.DOTALL | _re.IGNORECASE)

def _parse_cv_detail(source_filename: str) -> dict:
    """Lit le CV brut et extrait résumé, skills, langues, certifs."""
    # Try exact filename or stem match
    candidates_paths = list(RAW_DIR.glob(f"{source_filename}")) + \
                       list(RAW_DIR.glob(f"{source_filename}.txt")) + \
                       list(RAW_DIR.glob(f"*{source_filename.replace('.txt','')}*.txt"))
    if not candidates_paths:
        return {}
    text = candidates_paths[0].read_text(encoding="utf-8", errors="replace")

    summary_m = _RE_SUMMARY.search(text)
    summary   = summary_m.group(1).strip() if summary_m else ""

    tech  = [s.strip() for s in (_RE_TECH.search(text) or type('', (), {'group': lambda *a: ''})()).group(1).split(",") if s.strip()]
    meth  = [s.strip() for s in (_RE_METH.search(text) or type('', (), {'group': lambda *a: ''})()).group(1).split(",") if s.strip()]
    mgmt  = [s.strip() for s in (_RE_MAN.search(text)  or type('', (), {'group': lambda *a: ''})()).group(1).split(",") if s.strip()]

    langs = []
    lang_m = _RE_LANG_BLK.search(text)
    if lang_m:
        for line in lang_m.group(1).strip().splitlines():
            line = line.strip()
            if line:
                langs.append(line)

    certs = []
    cert_m = _RE_CERT_BLK.search(text)
    if cert_m:
        for line in cert_m.group(1).strip().splitlines():
            line = line.strip()
            if line and line.lower() != "none listed":
                certs.append(line)

    return {
        "summary":      summary,
        "skills_tech":  tech,
        "skills_meth":  meth,
        "skills_mgmt":  mgmt,
        "languages":    langs,
        "certifications": certs,
    }


@app.get("/candidates/{candidate_id}", tags=["rh"])
def get_candidate(candidate_id: str):
    """Retourne le profil complet d'un candidat (données CSV + contenu CV brut)."""
    if not CANDIDATES_FILE.exists():
        raise HTTPException(404, "Aucun candidat enregistré.")
    df = pd.read_csv(CANDIDATES_FILE)
    row = df[df["candidate_id"] == candidate_id]
    if row.empty:
        raise HTTPException(404, f"Candidat {candidate_id} introuvable.")
    data = row.iloc[0].fillna("").to_dict()
    detail = _parse_cv_detail(str(data.get("source_filename", "")))
    return {**data, **detail}


# ── Comments ─────────────────────────────────────────────────────

def _load_comments() -> dict:
    if not COMMENTS_FILE.exists():
        return {}
    return json.loads(COMMENTS_FILE.read_text(encoding="utf-8"))

def _save_comments(data: dict):
    COMMENTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

class CommentCreate(BaseModel):
    author: str
    text: str

class CommentUpdate(BaseModel):
    text: str

@app.get("/comments/{candidate_id}", tags=["rh"])
def get_comments(candidate_id: str):
    data = _load_comments()
    return data.get(candidate_id, [])

@app.post("/comments/{candidate_id}", tags=["rh"])
def add_comment(candidate_id: str, body: CommentCreate):
    data = _load_comments()
    thread = data.get(candidate_id, [])
    comment = {
        "id": uuid.uuid4().hex,
        "author": body.author,
        "text": body.text,
        "created_at": datetime.now().isoformat(),
        "updated_at": None,
    }
    thread.append(comment)
    data[candidate_id] = thread
    _save_comments(data)
    return comment

@app.patch("/comments/{candidate_id}/{comment_id}", tags=["rh"])
def update_comment(candidate_id: str, comment_id: str, body: CommentUpdate):
    data = _load_comments()
    thread = data.get(candidate_id, [])
    for c in thread:
        if c["id"] == comment_id:
            c["text"] = body.text
            c["updated_at"] = datetime.now().isoformat()
            data[candidate_id] = thread
            _save_comments(data)
            return c
    raise HTTPException(404, "Commentaire introuvable.")

@app.delete("/comments/{candidate_id}/{comment_id}", tags=["rh"])
def delete_comment(candidate_id: str, comment_id: str):
    data = _load_comments()
    thread = data.get(candidate_id, [])
    data[candidate_id] = [c for c in thread if c["id"] != comment_id]
    _save_comments(data)
    return {"deleted": comment_id}


# ── Analyse période ───────────────────────────────────────────────

FEATURE_LABELS = {
    "exp_per_year_of_age":    "Exp/âge normalisé",
    "avg_job_duration":       "Durée moy. poste",
    "education_level":        "Niveau études",
    "potential_score":        "Score potentiel",
    "junior_potential":       "Potentiel junior",
    "has_multiple_languages": "Plurilingue",
    "career_depth":           "Prof. carrière",
    "is_it":                  "Secteur IT",
    "field_match":            "Field match",
}

def _shap_narrative(shap_dict: dict, score: float, threshold: float, name: str = "") -> str:
    """Génère un résumé en langage naturel depuis les valeurs SHAP."""
    if not shap_dict:
        return "Pas de données d'explication disponibles."
    sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    positives = [(FEATURE_LABELS.get(k, k), v) for k, v in sorted_feats if v > 0.01][:3]
    negatives = [(FEATURE_LABELS.get(k, k), v) for k, v in sorted_feats if v < -0.01][:2]

    decision = "invité" if score >= threshold else "rejeté"
    parts = [f"Score de {round(score*100)}% — profil {decision}."]

    if positives:
        pos_str = ", ".join(f"{lbl}" for lbl, _ in positives)
        parts.append(f"Points forts : {pos_str}.")
    if negatives:
        neg_str = ", ".join(f"{lbl}" for lbl, _ in negatives)
        parts.append(f"Freins : {neg_str}.")
    return " ".join(parts)

@app.get("/analyse/period", tags=["rh"])
def analyse_period(
    start: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end:   Optional[str] = Query(None, description="YYYY-MM-DD"),
):
    """Analyse agrégée sur une période : SHAP, manques, distribution features."""
    if not CANDIDATES_FILE.exists():
        return {"candidates": [], "shap_aggregate": {}, "feature_comparison": {}}
    df = pd.read_csv(CANDIDATES_FILE)
    if "received_at" in df.columns:
        df["received_at"] = pd.to_datetime(df["received_at"], errors="coerce")
        if start:
            df = df[df["received_at"] >= pd.Timestamp(start)]
        if end:
            df = df[df["received_at"] <= pd.Timestamp(end) + pd.Timedelta(days=1)]

    df["score_num"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    df["thr_num"]   = pd.to_numeric(df["threshold_used"], errors="coerce").fillna(0.5)

    # Ensure optional columns exist (older CSV rows may not have them)
    for col in ["shap_json", "threshold_used", "sector", "name"]:
        if col not in df.columns:
            df[col] = None

    # Parse SHAP values
    shap_agg = {}
    shap_counts = {}
    for _, row in df.iterrows():
        try:
            sv = json.loads(row.get("shap_json") or "{}")
        except Exception:
            sv = {}
        for k, v in sv.items():
            shap_agg[k]    = shap_agg.get(k, 0) + abs(float(v))
            shap_counts[k] = shap_counts.get(k, 0) + 1

    shap_mean = {
        FEATURE_LABELS.get(k, k): round(shap_agg[k] / shap_counts[k], 4)
        for k in shap_agg if shap_counts[k] > 0
    }
    shap_mean = dict(sorted(shap_mean.items(), key=lambda x: x[1], reverse=True))

    # Feature comparison invited vs rejected
    feat_cols = [c for c in FEATURE_LABELS if c in df.columns]
    invited  = df[df["decision"] == "invite"]
    rejected = df[df["decision"] == "reject"]
    comparison = {}
    for c in feat_cols:
        inv_mean = round(float(pd.to_numeric(invited[c], errors="coerce").mean() or 0), 3)
        rej_mean = round(float(pd.to_numeric(rejected[c], errors="coerce").mean() or 0), 3)
        comparison[FEATURE_LABELS.get(c, c)] = {
            "invited_avg":  inv_mean,
            "rejected_avg": rej_mean,
            "gap": round(inv_mean - rej_mean, 3),
        }

    # Candidates summary for table
    candidates_out = df[["candidate_id","name","score","decision","threshold_used","received_at","sector","shap_json"]].copy()
    candidates_out["received_at"] = candidates_out["received_at"].astype(str)

    # Add narrative per candidate
    records = []
    for _, row in candidates_out.iterrows():
        try:
            sv = json.loads(row.get("shap_json") or "{}")
        except Exception:
            sv = {}
        narrative = _shap_narrative(sv, float(row["score"] or 0), float(row["threshold_used"] or 0.5))
        records.append({**row.fillna("").to_dict(), "narrative": narrative, "shap": sv})

    return {
        "total":              len(df),
        "invited":            int((df["decision"] == "invite").sum()),
        "rejected":           int((df["decision"] == "reject").sum()),
        "invite_rate":        round(int((df["decision"] == "invite").sum()) / max(len(df), 1) * 100, 1),
        "avg_score":          round(float(df["score_num"].mean()), 3) if len(df) else 0,
        "shap_aggregate":     shap_mean,
        "feature_comparison": comparison,
        "candidates":         records,
    }

@app.get("/candidates/{candidate_id}/explain", tags=["rh"])
def explain_candidate(candidate_id: str):
    """Retourne SHAP values + narrative pour un candidat."""
    if not CANDIDATES_FILE.exists():
        raise HTTPException(404, "Aucun candidat enregistré.")
    df = pd.read_csv(CANDIDATES_FILE)
    row = df[df["candidate_id"] == candidate_id]
    if row.empty:
        raise HTTPException(404, f"Candidat {candidate_id} introuvable.")
    r = row.iloc[0]
    try:
        sv = json.loads(r.get("shap_json") or "{}")
    except Exception:
        sv = {}
    score = float(r.get("score") or 0)
    thr   = float(r.get("threshold_used") or 0.5)

    # Avg of invited for comparison
    invited = df[df["decision"] == "invite"]
    feat_cols = [c for c in FEATURE_LABELS if c in df.columns]
    missing = []
    for c in feat_cols:
        candidate_val = float(pd.to_numeric(r.get(c), errors="coerce") or 0)
        invited_avg   = float(pd.to_numeric(invited[c], errors="coerce").mean() or 0)
        if invited_avg > 0 and candidate_val < invited_avg * 0.7:
            missing.append({
                "feature": FEATURE_LABELS.get(c, c),
                "candidate_value": round(candidate_val, 3),
                "invited_avg": round(invited_avg, 3),
                "gap_pct": round((invited_avg - candidate_val) / invited_avg * 100, 1),
            })

    labeled_shap = {FEATURE_LABELS.get(k, k): v for k, v in sv.items()}
    return {
        "shap": labeled_shap,
        "narrative": _shap_narrative(sv, score, thr),
        "missing": sorted(missing, key=lambda x: x["gap_pct"], reverse=True)[:4],
        "score": score,
        "threshold": thr,
    }


@app.get("/candidates/{candidate_id}/semantic", tags=["rh"])
def semantic_analysis(candidate_id: str):
    """
    Analyse sémantique LLM du CV brut.
    Retourne : équivalences compétences, trajectoire, gaps, score sémantique, signaux faibles.
    Nécessite ANTHROPIC_API_KEY dans l'environnement.
    """
    if _anthropic_client is None:
        raise HTTPException(503, "Client Anthropic non initialisé. Vérifiez ANTHROPIC_API_KEY.")

    raw_file = RAW_TEXTS_DIR / f"{candidate_id}.txt"
    if not raw_file.exists():
        raise HTTPException(404, f"Texte CV introuvable pour {candidate_id}. Soumettez d'abord le CV via /score.")

    cv_text = raw_file.read_text(encoding="utf-8", errors="ignore")[:12000]  # ~3k tokens max

    # Récupérer le profil ML existant pour contexte
    context_snippet = ""
    if CANDIDATES_FILE.exists():
        df = pd.read_csv(CANDIDATES_FILE)
        row = df[df["candidate_id"] == candidate_id]
        if not row.empty:
            r = row.iloc[0]
            context_snippet = (
                f"Score ML : {r.get('score', '?')} | Décision : {r.get('decision', '?')} | "
                f"Expérience : {r.get('years_experience', '?')} ans | "
                f"Secteur : {r.get('sector', '?')} | Éducation : {r.get('education_level', '?')}"
            )

    prompt = f"""Tu es un expert RH senior et psychologue du travail. Analyse ce CV de façon approfondie et bienveillante, en cherchant à révéler le potentiel réel du candidat au-delà des critères classiques.

CONTEXTE ML (pour information, ne pas reproduire) :
{context_snippet}

CV BRUT :
\"\"\"
{cv_text}
\"\"\"

Réponds UNIQUEMENT en JSON valide avec cette structure exacte :
{{
  "semantic_score": <entier 0-100, ta propre évaluation holistique>,
  "trajectory": "<2-3 phrases : sens de la carrière, cohérence, évolution, signaux de leadership>",
  "skill_equivalencies": [
    {{"stated": "<compétence telle qu'écrite>", "equivalent": "<traduction/équivalent reconnu>", "level": "<junior|confirmed|expert>"}}
  ],
  "career_gaps": [
    {{"period": "<période approximative>", "likely_reason": "<explication bienveillante>", "impact": "<faible|modéré|fort>"}}
  ],
  "hidden_gems": ["<signal positif non capturé par ML>"],
  "red_flags": ["<point de vigilance factuel, sans jugement>"],
  "recommendation": "<une phrase de recommandation pour le recruteur>"
}}

Sois factuel, bienveillant, et cherche les compétences transversales et le potentiel caché. Limite skill_equivalencies à 5 max, hidden_gems à 3 max, red_flags à 2 max."""

    try:
        message = _anthropic_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_response = message.content[0].text.strip()
        # Extraire le JSON (Claude peut ajouter du texte autour)
        import re
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Réponse LLM non parseable : {e}")
    except Exception as e:
        raise HTTPException(500, f"Erreur Anthropic API : {e}")

    return {
        "candidate_id": candidate_id,
        "model": "claude-opus-4-5",
        **result,
    }


@app.get("/analyse/spotcheck", tags=["rh"])
def spotcheck_rejected(n: int = Query(5, ge=1, le=20)):
    """
    Audit aléatoire de la pile rejetée.
    Retourne n candidats rejetés avec comparaison vs moyenne des invités.
    Objectif : détecter les profils potentiellement sous-scorés (faux négatifs).
    """
    if not CANDIDATES_FILE.exists():
        return {"spotcheck": [], "invited_averages": {}}

    df = pd.read_csv(CANDIDATES_FILE)
    feat_cols = [c for c in FEATURE_LABELS if c in df.columns]

    invited  = df[df["decision"] == "invite"]
    rejected = df[df["decision"] == "reject"]

    if rejected.empty:
        return {"spotcheck": [], "invited_averages": {}, "message": "Aucun candidat rejeté."}

    # Moyennes des invités
    invited_avgs = {}
    for c in feat_cols:
        val = pd.to_numeric(invited[c], errors="coerce").mean()
        if not pd.isna(val):
            invited_avgs[FEATURE_LABELS.get(c, c)] = round(float(val), 3)

    # Sélection aléatoire de n rejetés
    sample = rejected.sample(min(n, len(rejected)), random_state=None)

    spotcheck_out = []
    for _, row in sample.iterrows():
        score = float(row.get("score") or 0)
        thr   = float(row.get("threshold_used") or 0.5)
        gap   = round(thr - score, 3)  # distance au seuil (positif = sous le seuil)

        # Features proches des moyennes invités : signaux de sous-scoring
        near_invite = []
        for c in feat_cols:
            cand_val = float(pd.to_numeric(row.get(c), errors="coerce") or 0)
            inv_avg  = float(pd.to_numeric(invited[c], errors="coerce").mean() or 0)
            if inv_avg > 0 and cand_val >= inv_avg * 0.85:
                near_invite.append({
                    "feature": FEATURE_LABELS.get(c, c),
                    "candidate_value": round(cand_val, 3),
                    "invited_avg": round(inv_avg, 3),
                    "pct_of_avg": round(cand_val / inv_avg * 100, 1),
                })

        try:
            sv = json.loads(row.get("shap_json") or "{}")
        except Exception:
            sv = {}
        labeled_shap = {FEATURE_LABELS.get(k, k): v for k, v in sv.items()}

        spotcheck_out.append({
            "candidate_id":   str(row.get("candidate_id", "")),
            "name":           str(row.get("name") or "—"),
            "sector":         str(row.get("sector") or "—"),
            "score":          score,
            "threshold":      thr,
            "gap_to_threshold": gap,
            "received_at":    str(row.get("received_at") or ""),
            "near_invite_features": sorted(near_invite, key=lambda x: x["pct_of_avg"], reverse=True)[:4],
            "shap":           labeled_shap,
            "narrative":      _shap_narrative(sv, score, thr, str(row.get("name") or "")),
            "suspicious":     gap < 0.05,  # très proche du seuil → potentiel faux négatif
        })

    # Trier par proximité au seuil (plus suspect en premier)
    spotcheck_out.sort(key=lambda x: x["gap_to_threshold"])

    return {
        "n_rejected_total": len(rejected),
        "n_sampled":        len(spotcheck_out),
        "invited_averages": invited_avgs,
        "spotcheck":        spotcheck_out,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
