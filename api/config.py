"""
config.py — Chemins et constantes partagées dans toute l'API.
"""

import sys
from pathlib import Path

# Racine du projet (dossier Automatic_CV)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Répertoires
MODELS_DIR       = ROOT / "models"
PROCESSED_DIR    = ROOT / "data" / "processed"
RAW_TEXTS_DIR    = PROCESSED_DIR / "raw_texts"
RAW_DIR          = ROOT / "data" / "raw"
CANDIDATES_FILE  = PROCESSED_DIR / "candidates_live.csv"
COMMENTS_FILE    = PROCESSED_DIR / "comments.json"
CONFIG_DIR       = ROOT / "config"
ELIMINATORY_FILE = CONFIG_DIR / "eliminatory_criteria.json"

# Créer les dossiers manquants au démarrage
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_TEXTS_DIR.mkdir(parents=True, exist_ok=True)

# Champs exportés dans le CSV candidats
CANDIDATE_FIELDS = [
    "candidate_id", "received_at", "source_filename", "name", "email", "phone",
    "gender", "age", "sector", "target_role", "years_experience", "education_level",
    "score", "decision", "threshold_used", "status", "shap_json",
    "priority_rank", "eliminated_reason",
]

# Labels lisibles pour les features SHAP
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
