# CV-Intelligence

Système de tri automatisé de CVs par IA pour TechCore Liège (contexte fictif).  
Conformité RGPD et EU AI Act (haut risque, Annexe III).

---

## État d'avancement

| Phase | Objectif | Statut |
|---|---|---|
| 1 | Infrastructure n8n + API FastAPI | ✅ Terminé |
| 2 | Parsing, anonymisation, feature engineering | ✅ Terminé |
| 3 | Entraînement du modèle ML + audit biais + SHAP | ✅ Terminé |
| 4 | Dashboard RH + conformité + intégration n8n → ML | 🔄 En cours |
| 5 | Tests, client pilote, documentation finale | 📅 Planifié |

---

## Structure du projet

```
cv-intelligence/
├── pipeline_ml/
│   ├── run.py               # Interface de lancement du pipeline (menu + CLI)
│   ├── parse_cv.py          # Parsing + pseudonymisation des CV .txt
│   ├── feature_engineering.py  # Features composites (exp_edu_score, tech_per_year…)
│   ├── train.py             # Entraînement et évaluation du modèle ML
│   ├── audit.py             # Audit biais (Fairlearn) + SHAP
│   ├── pseudo_labels.py     # Génération de pseudo-labels (bootstrapping initial)
│   └── student_labels.csv   # Labels réels recruteur (200 CVs)
│
├── data/
│   ├── raw/                 # CV bruts .txt (ignoré par Git — RGPD)
│   └── processed/           # features.csv + identities.csv (ignoré par Git)
│
├── models/                  # model.pkl, scaler.pkl, feature_cols.pkl (ignoré par Git)
├── reports/                 # evaluation.txt, audit.txt (ignoré par Git)
│
├── docs/
│   ├── roadmap.md
│   ├── entreprise_fictive.md
│   └── checkup/
│       ├── n8n.md           # Architecture pipeline d'ingestion
│       └── model.md         # Résultats et analyse du modèle ML
│
└── api/                     # FastAPI (Phase 1)
```

---

## Prérequis

- Python 3.11+
- Packages : `scikit-learn`, `pandas`, `joblib`, `numpy`, `shap`, `fairlearn`, `xgboost`

```bash
pip install scikit-learn pandas joblib numpy shap fairlearn xgboost
```

---

## Utilisation — Pipeline ML

### Lancement rapide (pipeline complet)

```bash
python pipeline_ml/run.py full
```

### Menu interactif

```bash
python pipeline_ml/run.py
```

Étapes disponibles :
```
[1] Parsing + pseudonymisation (labels réels)
[2] Feature engineering
[3] Entraînement du modèle
[4] Audit biais + SHAP
[5] Pipeline complet (1→4)
```

---

## Détail des étapes

### Étape 1 — Parser les CV

```bash
python pipeline_ml/parse_cv.py
```

Lit les fichiers `.txt` dans `data/raw/`, charge les labels depuis `pipeline_ml/student_labels.csv`.  
Produit deux fichiers dans `data/processed/` :

| Fichier | Contenu |
|---|---|
| `features.csv` | Features anonymisées + label + heuristic_score |
| `identities.csv` | Données sensibles (nom, email, téléphone, genre, âge) liées par `cv_id` |

> `identities.csv` ne doit **jamais** être passé au modèle ML.

### Étape 2 — Feature engineering

```bash
python pipeline_ml/feature_engineering.py
```

Enrichit `features.csv` avec 9 features composites :

| Feature | Signal |
|---|---|
| `exp_edu_score` | Séniorité × diplôme — r=+0.364 |
| `career_depth` | Expérience longue et stable — r=+0.257 |
| `tech_per_year` | Détecte les CV gonflés en skills — r=−0.243 |
| `multilingual_score` | Score langues composite (nb_languages + bonus anglais) |
| `log_years_exp`, `cert_density`, `method_tech_ratio`, `is_it`, `is_finance` | Autres |

### Étape 3 — Entraîner le modèle

```bash
python pipeline_ml/train.py
```

Compare Régression Logistique, Random Forest et XGBoost.  
Résultats actuels (200 CVs, labels réels, Avril 2026) :

| Modèle | F1 (test) | ROC-AUC |
|---|---|---|
| **Régression Logistique** | **0.621** | **0.837** |
| Random Forest | 0.267 | 0.727 |
| XGBoost | 0.333 | 0.640 |

### Étape 4 — Audit biais + SHAP

```bash
python pipeline_ml/audit.py
```

Produit `reports/audit.txt` avec :
- Analyse structurelle du dataset (distribution labels, features)
- Parité genre : Disparate Impact Ratio (cible ≥ 0.80)
- Parité âge : DI par groupe (junior/senior)
- Fairlearn : Demographic Parity Difference, Equal Opportunity Difference
- SHAP : importance globale des features + top features par profil type

---

## Architecture RGPD

```
CV brut (.txt)
   │
   ├─ parse_cv.py
   │     ├─ identities.csv  ←  nom, email, téléphone, genre, âge  (jamais au modèle)
   │     └─  features.csv   ←  19 features anonymisées + label     (input ML)
   │
   ├─ feature_engineering.py  →  features.csv enrichi (28 colonnes)
   │
   ├─ train.py              →  models/model.pkl
   │
   └─ audit.py              →  reports/audit.txt
```

Chaque candidat reçoit un `cv_id` UUID déterministe à l'ingestion.  
Les deux fichiers se joignent uniquement via `cv_id`, jamais via des données nominatives.

---

## Conformité EU AI Act

Le recrutement est classé **usage à haut risque** (Annexe III). Obligations appliquées :

- Décision finale toujours humaine — le modèle produit un score, pas une décision
- Aucun rejet définitif sans révision humaine tracée
- Données sensibles (`gender`, `age`) exclues du modèle — monitoring biais uniquement
- `heuristic_score` conservé pour contestation et révision humaine
- Explicabilité SHAP disponible par candidat (`pipeline_ml/audit.py`)
- Audit biais Fairlearn (DI, DPD, EOD) sur chaque run

---

## Infrastructure Phase 1 (n8n)

Pipeline d'ingestion automatique opérationnel sur `https://n8n.lony.app`.  
Voir `docs/checkup/n8n.md` pour le détail de l'architecture.

---

## Documentation

| Fichier | Contenu |
|---|---|
| `docs/roadmap.md` | Roadmap complète du projet |
| `docs/entreprise_fictive.md` | Contexte TechCore + logique du bootstrapping |
| `docs/checkup/n8n.md` | Architecture pipeline n8n + API |
| `docs/checkup/model.md` | Résultats ML, features, interprétation, limites |
