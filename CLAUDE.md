# CV-Intelligence — CLAUDE.md

Système de tri automatisé de CVs par ML pour TechCore Liège (contexte fictif).
Conformité RGPD + EU AI Act (haut risque, Annexe III).

## Branches

- `develop` — branche active, tout le travail va ici
- `main` — stable uniquement

## Stack

| Couche | Techno |
|---|---|
| Pipeline ML | Python 3.11, scikit-learn, pandas, shap, groq |
| MLOps | MLflow (tracking local dans `mlruns/`) |
| API | FastAPI + SQLAlchemy + PostgreSQL |
| Frontend | React 19, Vite, Tailwind CSS, dnd-kit, recharts |
| Ingestion | n8n sur `https://n8n.lony.app` |

## Structure

```
pipeline_ml/
  core/
    p01_parse.py       # Parsing CV — regex (.txt) + LLM Groq (.pdf/.docx/.txt)
    p02_features.py    # Feature engineering (9 features composites)
    p03_analysis.py    # EDA
    p04_train.py       # Entraînement LR + Grid Search + MLflow tracking
    p06_audit.py       # Audit biais SHAP + fairness + MLflow tracking
    p07_labeling.py    # Labeling manuel
  tests/
    test_pipeline.py   # Tests d'intégration (imports, MLflow, Groq, regex)
  run.py               # Menu interactif (0→6 + pipeline complet)
  requirements_ml.txt

api/
  main.py              # FastAPI — endpoints scoring, candidats, jobs, auth
  requirements_api.txt

frontend/              # React — ATS dashboard
  src/
  package.json

data/                  # Ignoré par git (RGPD)
  raw/                 # CVs bruts .txt / .pdf / .docx
  processed/           # features.csv + identities.csv

models/                # Ignoré par git
  model.pkl, scaler.pkl, feature_cols.pkl, threshold.pkl, threshold_junior.pkl
  mlflow_run_id.txt    # Run ID du dernier entraînement (lien p04 → p06)

mlruns/                # Ignoré par git — tracking MLflow local
```

## Variables d'environnement (.env)

```
GROQ_API_KEY=...       # Parsing LLM (llama-3.3-70b-versatile)
DATABASE_URL=...       # PostgreSQL pour l'API
SECRET_KEY=...         # JWT
```

## Lancer le pipeline ML

```bash
# Installation
pip install -r pipeline_ml/requirements_ml.txt

# Menu interactif
python pipeline_ml/run.py

# Pipeline complet en une commande
python -c "from pipeline_ml.run import run_full; run_full()"

# Étapes individuelles
python -m pipeline_ml.core.p01_parse              # parsing regex (défaut)
python -m pipeline_ml.core.p01_parse --parser llm # parsing Groq (PDF/DOCX)
python -m pipeline_ml.core.p04_train              # entraînement + MLflow
python -m pipeline_ml.core.p06_audit              # audit biais + MLflow

# Voir les runs MLflow
mlflow ui   # → http://localhost:5000
```

## Lancer les tests

```bash
# Tests d'intégration complets (imports, MLflow, extraction PDF/DOCX, regex, Groq×5)
python pipeline_ml/tests/test_pipeline.py

# Ou via pytest
pytest pipeline_ml/tests/test_pipeline.py -v
```

Le test Groq nécessite `GROQ_API_KEY` et des fichiers dans `data/raw/`.
Les autres tests fonctionnent sans données.

## Lancer l'API

```bash
docker compose up api -d
# ou
uvicorn api.main:app --reload --port 8000
```

## Lancer le frontend

```bash
cd frontend
npm install
npm run dev   # → http://localhost:5173
# Le proxy Vite redirige /auth /candidates /jobs /interviews /stats /score vers :8000
```

## Conventions

- Composants React en PascalCase, hooks en `use*`
- Appels API uniquement via `src/lib/api.js`
- Tailwind pour le style, pas de CSS custom sauf `index.css`
- dnd-kit pour tout drag & drop, recharts pour tous les graphiques
- Pas de state management externe (React Context suffit)
- Séparation RGPD stricte : `identities.csv` jamais passé au modèle

## Roadmap active

- `docs/roadmap.md` — roadmap ML (phases 1-5)
- `docs/mlops_and_ocr_roadmap.md` — MLflow, DVC, GLM-OCR, LLM parsing
- `docs/roadmap_frontend.md` — 7 phases ATS (Kanban, Scorecards, IA co-pilot…)
