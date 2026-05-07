"""
routers/stats.py — Dashboard stats, analyse période et spot-check.
"""

import json
import logging
from typing import Optional

import pandas as pd
from datetime import datetime
from fastapi import APIRouter, Query

from api.config import CANDIDATES_FILE, FEATURE_LABELS
from api.database import get_db, Candidate as CandidateModel

router = APIRouter(tags=["stats"])
logger = logging.getLogger("cv_api")


# ── Helpers narratif SHAP ────────────────────────────────────────────
def shap_narrative(shap_dict: dict, score: float, threshold: float, name: str = "") -> str:
    if not shap_dict:
        return "Pas de données d'explication disponibles."
    sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    positives = [(FEATURE_LABELS.get(k, k), v) for k, v in sorted_feats if v > 0.01][:3]
    negatives = [(FEATURE_LABELS.get(k, k), v) for k, v in sorted_feats if v < -0.01][:2]

    decision = "invité" if score >= threshold else "rejeté"
    parts = [f"Score de {round(score*100)}% — profil {decision}."]
    if positives:
        parts.append(f"Points forts : {', '.join(lbl for lbl, _ in positives)}.")
    if negatives:
        parts.append(f"Freins : {', '.join(lbl for lbl, _ in negatives)}.")
    return " ".join(parts)


def _load_df() -> Optional[pd.DataFrame]:
    """Charge les candidats depuis DB ou CSV."""
    try:
        with get_db() as db:
            rows = db.query(CandidateModel).all()
            if rows:
                return pd.DataFrame([
                    {c.name: getattr(r, c.name) for c in r.__table__.columns}
                    for r in rows
                ])
    except Exception as e:
        logger.warning(f"DB read for stats failed ({e})")
    if CANDIDATES_FILE.exists():
        return pd.read_csv(CANDIDATES_FILE)
    return None


# ── GET /stats ───────────────────────────────────────────────────────
@router.get("/stats")
def get_stats():
    df = _load_df()
    if df is None or df.empty:
        return {"total": 0, "invited": 0, "rejected": 0, "invite_rate": 0,
                "today": 0, "borderline": 0, "pending_review": 0, "avg_score": 0,
                "interesting_rejected": [], "borderline_list": [], "by_sector": {}}

    if "status" not in df.columns:
        df["status"] = "inbox"
    df["score_num"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    df["thr_num"]   = pd.to_numeric(df["threshold_used"], errors="coerce").fillna(0.5)

    today    = datetime.now().date().isoformat()
    total    = len(df)
    invited  = int((df["decision"] == "invite").sum())
    rejected = int((df["decision"] == "reject").sum())

    df["gap"]  = (df["score_num"] - df["thr_num"]).abs()
    borderline = int((df["gap"] <= 0.08).sum())
    pending    = int(df["status"].isin(["inbox", "review"]).sum())
    today_count = int(df["received_at"].astype(str).str.startswith(today).sum())
    avg_score   = round(float(df["score_num"].mean()), 3) if total else 0

    interesting_rejected = df[
        (df["decision"] == "reject") & (df["score_num"] >= 0.3)
    ][["candidate_id", "name", "score", "target_role", "sector"]].head(5).fillna("").to_dict(orient="records")

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
            str(s): {"total": len(g), "invited": int((g["decision"] == "invite").sum())}
            for s, g in df.groupby("sector", dropna=True)
        },
    }


# ── GET /analyse/period ──────────────────────────────────────────────
@router.get("/analyse/period")
def analyse_period(
    start: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end:   Optional[str] = Query(None, description="YYYY-MM-DD"),
):
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
    for col in ["shap_json", "threshold_used", "sector", "name"]:
        if col not in df.columns:
            df[col] = None

    shap_agg, shap_counts = {}, {}
    for _, row in df.iterrows():
        try:
            sv = json.loads(row.get("shap_json") or "{}")
        except Exception:
            sv = {}
        for k, v in sv.items():
            shap_agg[k]    = shap_agg.get(k, 0) + abs(float(v))
            shap_counts[k] = shap_counts.get(k, 0) + 1

    shap_mean = dict(sorted({
        FEATURE_LABELS.get(k, k): round(shap_agg[k] / shap_counts[k], 4)
        for k in shap_agg if shap_counts[k] > 0
    }.items(), key=lambda x: x[1], reverse=True))

    feat_cols = [c for c in FEATURE_LABELS if c in df.columns]
    invited, rejected_df = df[df["decision"] == "invite"], df[df["decision"] == "reject"]
    comparison = {}
    for c in feat_cols:
        inv_mean = round(float(pd.to_numeric(invited[c],    errors="coerce").mean() or 0), 3)
        rej_mean = round(float(pd.to_numeric(rejected_df[c], errors="coerce").mean() or 0), 3)
        comparison[FEATURE_LABELS.get(c, c)] = {
            "invited_avg": inv_mean, "rejected_avg": rej_mean,
            "gap": round(inv_mean - rej_mean, 3),
        }

    candidates_out = df[["candidate_id","name","score","decision","threshold_used","received_at","sector","shap_json"]].copy()
    candidates_out["received_at"] = candidates_out["received_at"].astype(str)
    records = []
    for _, row in candidates_out.iterrows():
        try:
            sv = json.loads(row.get("shap_json") or "{}")
        except Exception:
            sv = {}
        records.append({
            **row.fillna("").to_dict(),
            "narrative": shap_narrative(sv, float(row["score"] or 0), float(row["threshold_used"] or 0.5)),
            "shap": sv,
        })

    return {
        "total":              len(df),
        "invited":            int((df["decision"] == "invite").sum()),
        "rejected":           int((df["decision"] == "reject").sum()),
        "invite_rate":        round(int((df["decision"] == "invite").sum()) / max(len(df),1) * 100, 1),
        "avg_score":          round(float(df["score_num"].mean()), 3) if len(df) else 0,
        "shap_aggregate":     shap_mean,
        "feature_comparison": comparison,
        "candidates":         records,
    }


# ── GET /analyse/spotcheck ───────────────────────────────────────────
@router.get("/analyse/spotcheck")
def spotcheck_rejected(n: int = Query(5, ge=1, le=20)):
    if not CANDIDATES_FILE.exists():
        return {"spotcheck": [], "invited_averages": {}}

    df = pd.read_csv(CANDIDATES_FILE)
    feat_cols = [c for c in FEATURE_LABELS if c in df.columns]
    invited   = df[df["decision"] == "invite"]
    rejected  = df[df["decision"] == "reject"]

    if rejected.empty:
        return {"spotcheck": [], "invited_averages": {}, "message": "Aucun candidat rejeté."}

    invited_avgs = {
        FEATURE_LABELS.get(c, c): round(float(pd.to_numeric(invited[c], errors="coerce").mean()), 3)
        for c in feat_cols
        if not pd.isna(pd.to_numeric(invited[c], errors="coerce").mean())
    }

    sample = rejected.sample(min(n, len(rejected)), random_state=None)
    spotcheck_out = []
    for _, row in sample.iterrows():
        score = float(row.get("score") or 0)
        thr   = float(row.get("threshold_used") or 0.5)
        gap   = round(thr - score, 3)

        near_invite = []
        for c in feat_cols:
            cand_val = float(pd.to_numeric(row.get(c), errors="coerce") or 0)
            inv_avg  = float(pd.to_numeric(invited[c], errors="coerce").mean() or 0)
            if inv_avg > 0 and cand_val >= inv_avg * 0.85:
                near_invite.append({
                    "feature":         FEATURE_LABELS.get(c, c),
                    "candidate_value": round(cand_val, 3),
                    "invited_avg":     round(inv_avg, 3),
                    "pct_of_avg":      round(cand_val / inv_avg * 100, 1),
                })

        try:
            sv = json.loads(row.get("shap_json") or "{}")
        except Exception:
            sv = {}
        labeled_shap = {FEATURE_LABELS.get(k, k): v for k, v in sv.items()}

        spotcheck_out.append({
            "candidate_id":          str(row.get("candidate_id", "")),
            "name":                  str(row.get("name") or "—"),
            "sector":                str(row.get("sector") or "—"),
            "score":                 score,
            "threshold":             thr,
            "gap_to_threshold":      gap,
            "received_at":           str(row.get("received_at") or ""),
            "near_invite_features":  sorted(near_invite, key=lambda x: x["pct_of_avg"], reverse=True)[:4],
            "shap":                  labeled_shap,
            "narrative":             shap_narrative(sv, score, thr, str(row.get("name") or "")),
            "suspicious":            gap < 0.05,
        })

    spotcheck_out.sort(key=lambda x: x["gap_to_threshold"])
    return {
        "n_rejected_total": len(rejected),
        "n_sampled":        len(spotcheck_out),
        "invited_averages": invited_avgs,
        "spotcheck":        spotcheck_out,
    }
