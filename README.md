# 🤖 CV-Intelligence — Système de Tri Automatisé de CV par IA

> Pipeline bout-en-bout : réception automatique → parsing NLP → scoring objectif → ML maison → dashboard RH

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat&logo=fastapi&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?style=flat&logo=supabase&logoColor=white)
![n8n](https://img.shields.io/badge/n8n-automation-EA4B71?style=flat&logo=n8n&logoColor=white)
![React](https://img.shields.io/badge/Dashboard-React+Vite-61DAFB?style=flat&logo=react&logoColor=white)
![License](https://img.shields.io/badge/License-Proprietary-red?style=flat)

---

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Prérequis](#prérequis)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Démarrage](#démarrage)
7. [Pipeline de traitement](#pipeline-de-traitement)
8. [Automatisation n8n](#automatisation-n8n)
9. [Dashboard](#dashboard)
10. [API Reference](#api-reference)
11. [Machine Learning](#machine-learning)
12. [RGPD & Conformité](#rgpd--conformité)
13. [Tests](#tests)
14. [Roadmap](#roadmap)
15. [Contributing](#contributing)

---

## Vue d'ensemble

CV-Intelligence est une plateforme SaaS B2B d'analyse automatisée de CVs combinant :

- **Scoring objectif universel** — règles métier configurables par fiche de poste (JSON)
- **ML maison par entreprise** — modèle XGBoost entraîné sur les recrutements historiques du client
- **Automatisation n8n** — réception automatique depuis email, formulaire web, ou upload direct
- **Audit de biais intégré** — conformité EU AI Act via Fairlearn + SHAP
- **Dashboard temps réel** — interface RH avec classement explicable et filtres avancés

### Flux en une phrase

```
Email / Form / Upload → n8n → API Python → Parser → NER → Anonymisation → Score → ML → Supabase → Dashboard RH
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        SOURCES D'ENTRÉE                          │
│    📧 Email (IMAP)    🌐 Webhook Form    📁 Upload API           │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION n8n                           │
│   Trigger → Extract → Call API → Branch → Notify → Store        │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    PIPELINE PYTHON (FastAPI)                     │
│                                                                  │
│   01 INGESTION      pdfminer / python-docx / Tika + OCR         │
│        ↓                                                         │
│   02 NLP/NER        spaCy fr_core_news_lg (fine-tuné)           │
│        ↓                                                         │
│   03 ANONYMISATION  Microsoft Presidio → JSON pseudonymisé      │
│        ↓                                                         │
│   04 SCORING        Moteur règles métier → score /100           │
│        ↓                                                         │
│   05 ML MAISON      XGBoost par entreprise → fit score          │
│        ↓                                                         │
│   06 BIAIS AUDIT    Fairlearn + SHAP → rapport explicable       │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     SUPABASE (PostgreSQL)                        │
│   Tables : candidates · jobs · audit_logs                        │
│   Extension pgvector → stockage embeddings CV (remplace Qdrant) │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    DASHBOARD REACT (Vite)                        │
│   Classement · Scores détaillés · Audit biais · Export          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Prérequis

Chaque développeur installe uniquement :

| Outil | Version | Usage |
|-------|---------|-------|
| Python | 3.11+ | Backend + pipeline ML |
| Node.js | 18+ | Dashboard React |
| n8n | 1.0+ | Automatisation (`npx n8n`) |
| Tesseract OCR | 5+ | Parsing CVs scannés |

> **Pas de Docker nécessaire.** La base de données est hébergée sur Supabase et partagée entre tous les développeurs via l'URL de connexion dans `.env`.

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-org/cv-intelligence.git
cd cv-intelligence
```

### 2. Backend Python

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt

# Télécharger le modèle spaCy français
python -m spacy download fr_core_news_lg

# Appliquer les migrations sur Supabase
alembic upgrade head
```

### 3. Dashboard React

```bash
cd dashboard
npm install
```

### 4. n8n

```bash
npx n8n
# Interface accessible sur http://localhost:5678
# Importer le workflow : n8n/workflows/cv_pipeline_main.json
```

### 5. Modèles ML — sentence-transformers (optionnel, phase 2)

```bash
cd ml
python scripts/download_models.py
# Télécharge paraphrase-multilingual-mpnet-base-v2 depuis Hugging Face (~420 MB)
```

---

## Configuration

### .env — à faire par chaque développeur

```bash
cp .env.example .env
# Remplir avec l'URL Supabase partagée (communiquée par le lead dev)
```

```env
# ── Base de données Supabase (partagée entre tous les devs) ──────
DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres

# ── Sécurité ─────────────────────────────────────────────────────
SECRET_KEY=changez-moi-en-production
API_KEY_HEADER=X-API-Key

# ── n8n ──────────────────────────────────────────────────────────
N8N_WEBHOOK_SECRET=votre_secret_ici

# ── Optionnel : pré-annotation NER via LLM ───────────────────────
OPENAI_API_KEY=sk-...

# ── RGPD ─────────────────────────────────────────────────────────
DATA_RETENTION_DAYS=365
ANONYMIZE_AFTER_DAYS=30
```

> ⚠️ Le fichier `.env` ne doit **jamais** être commité sur Git. Il est listé dans `.gitignore`.

### Supabase — activation pgvector

Dans le dashboard Supabase, aller dans **Database → Extensions** et activer `vector`. Cela remplace Qdrant pour le stockage des embeddings de CVs.

```sql
-- À exécuter une seule fois dans l'éditeur SQL Supabase
create extension if not exists vector;
```

### Configuration d'une fiche de poste

Chaque poste est défini dans `config/jobs/` :

```json
{
  "job_id": "lead-dev-python-2026",
  "title": "Lead Developer Python",
  "seuil_minimum": 60,
  "criteres": {
    "competences_requises": {
      "poids": 0.35,
      "must_have": ["Python", "Git", "SQL"],
      "nice_to_have": ["Docker", "FastAPI", "Kubernetes"]
    },
    "experience_annees": { "poids": 0.25, "minimum": 4 },
    "formation":         { "poids": 0.20, "niveau_minimum": "Bac+3" },
    "mobilite":          { "poids": 0.10, "zones": ["Paris", "Lyon", "Remote"] },
    "langue":            { "poids": 0.10, "requis": ["Français", "Anglais"] }
  }
}
```

---

## Démarrage

```bash
# Terminal 1 — API Backend
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Dashboard
cd dashboard
npm run dev

# Terminal 3 — n8n
npx n8n
```

**URLs locales :**

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Swagger (docs API) | http://localhost:8000/docs |
| Dashboard | http://localhost:5173 |
| n8n | http://localhost:5678 |

---

## Pipeline de traitement

### Soumettre un CV manuellement

```bash
curl -X POST http://localhost:8000/api/v1/candidates \
  -H "X-API-Key: votre_cle" \
  -F "file=@cv_candidat.pdf" \
  -F "job_id=lead-dev-python-2026"
```

### Résultat JSON

```json
{
  "candidate_id": "cand_8f3a92b1",
  "job_id": "lead-dev-python-2026",
  "status": "completed",
  "score_objectif": 78,
  "score_ml_fit": 82,
  "score_final": 80,
  "ranking": 2,
  "decision_suggeree": "ENTRETIEN",
  "explications": [
    "Fort sur Python, Docker et architecture microservices",
    "5 ans d'expérience, dépasse le minimum requis (4 ans)",
    "Manque Kubernetes — formation courte envisageable"
  ],
  "biais_corrige": false,
  "rgpd_compliant": true,
  "processed_at": "2026-03-05T14:32:11Z"
}
```

---

## Automatisation n8n

### Importer le workflow principal

Dans l'interface n8n : **Workflows → Import from file → `n8n/workflows/cv_pipeline_main.json`**

### Sources configurées

| Source | Déclencheur |
|--------|-------------|
| Email IMAP | Polling toutes les 2 minutes |
| Formulaire web | Webhook POST |
| Upload direct | Webhook POST multipart |

### Logique de routage

```
CV reçu
  │
  ├─ score_final ≥ seuil ET score_ml ≥ 0.70  →  shortlist   + Slack "🟢 Profil fort"
  ├─ score_final ≥ seuil ET score_ml < 0.70  →  candidates  + Slack "🟡 À examiner"
  ├─ score_final < seuil                      →  rejected    + email candidat (optionnel)
  └─ erreur parsing                           →  manual_review + alerte recruteur
```

---

## Dashboard

Interface React accessible sur `http://localhost:5173` :

- **Vue globale** — volume de CVs, délai moyen, taux de shortlist
- **Tableau classement** — trié par score final, filtrable par poste / date / statut
- **Fiche candidat** — score détaillé par dimension + explication SHAP + actions RH
- **Rapport biais** — Disparate Impact Ratio par groupe, évolution dans le temps
- **Export** — shortlist en CSV ou PDF

---

## API Reference

Documentation interactive complète sur `/docs` (Swagger UI).

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/v1/candidates` | POST | Soumettre un CV |
| `/api/v1/candidates/{id}` | GET | Résultat d'un candidat |
| `/api/v1/candidates` | GET | Liste paginée |
| `/api/v1/jobs` | GET / POST | Gestion des fiches de poste |
| `/api/v1/jobs/{id}/ranking` | GET | Classement pour un poste |
| `/api/v1/ml/train` | POST | Lancer un re-entraînement ML |
| `/api/v1/ml/status` | GET | Statut du modèle actif |
| `/api/v1/bias-report` | GET | Rapport Fairlearn |
| `/api/v1/export/{job_id}` | GET | Export CSV / PDF |

---

## Machine Learning

### Entraîner le modèle ML maison

```bash
cd ml
python train.py \
  --company-id entreprise_abc \
  --data-path data/recrutements_historiques.csv \
  --min-samples 200 \
  --output models/entreprise_abc/
```

### Fine-tuner le NER spaCy

```bash
cd ml

# Préparer les données (export Label Studio)
python scripts/prepare_ner_data.py \
  --input data/annotations_label_studio.json \
  --output data/ner_train.spacy

# Entraîner
python -m spacy train config/ner_config.cfg \
  --output models/ner_cv_fr/ \
  --paths.train data/ner_train.spacy \
  --paths.dev data/ner_dev.spacy
```

### Format du CSV d'historique client

```
cv_text, competences, experience_annees, diplome_niveau, score_objectif, label
"Développeur Python...", "Python;Docker;SQL", 5, "Bac+5", 78, 1
"Graphiste freelance...", "Photoshop;Illustrator", 3, "Bac+3", 42, 0
```

`label` : `1` = recrutement réussi, `0` = refusé ou échec dans le poste.

### Volume de données requis

| Modèle | Minimum | Recommandé |
|--------|---------|------------|
| NER fine-tuning (spaCy) | 300 CVs annotés | 500+ |
| ML fit culturel (XGBoost) | 200 recrutements historiques | 500+ |
| Vectorisation (sentence-transformers) | Aucune donnée | — |
| Anonymisation (Presidio) | Aucune donnée | — |

---

## RGPD & Conformité

- **Pseudonymisation** — données personnelles hachées dès l'ingestion (Presidio)
- **Minimisation** — seules les données nécessaires au scoring sont conservées
- **Droit à l'oubli** — `DELETE /api/v1/candidates/{id}` supprime toutes les données liées
- **Durée de conservation** — configurable via `DATA_RETENTION_DAYS` dans `.env`
- **EU AI Act** — système classé haut risque (Annexe III) : explicabilité SHAP, audit Fairlearn, log immuable, supervision humaine obligatoire avant décision finale

---

## Tests

```bash
cd backend
source venv/bin/activate

# Tous les tests
pytest

# Avec rapport de coverage
pytest --cov=app --cov-report=html

# Module spécifique
pytest tests/test_scoring.py -v

# Tests d'intégration (nécessite connexion Supabase active)
pytest tests/integration/ -v
```

---

## Roadmap

| Période | Objectif | Statut |
|---------|----------|--------|
| Mars 2026 | Parser multi-format + NER baseline + structure Supabase | 🔄 En cours |
| Avril 2026 | Anonymisation + Scoring v1 + API + n8n + Dashboard v1 | 📅 Planifié |
| Mai 2026 | ML XGBoost v1 + Fine-tuning NER + Fairlearn | 📅 Planifié |
| Juin 2026 | Tests + client pilote + documentation finale | 📅 Planifié |

---

## Contributing

```bash
# Créer une branche feature
git checkout -b feature/nom-de-la-feature

# Conventions de commit
git commit -m "feat(parser): add OCR fallback for scanned PDFs"
git commit -m "fix(scoring): correct weight normalization"
git commit -m "docs(api): update endpoint documentation"

# Push et Pull Request vers main
git push origin feature/nom-de-la-feature
```

| Préfixe | Usage |
|---------|-------|
| `feat` | Nouvelle fonctionnalité |
| `fix` | Correction de bug |
| `docs` | Documentation |
| `test` | Ajout / modification de tests |
| `refactor` | Refactoring sans changement fonctionnel |
| `chore` | Maintenance, dépendances |

> Toute PR doit passer les tests (`pytest`) avant d'être mergée sur `main`.

---

*Dernière mise à jour : Mars 2026*