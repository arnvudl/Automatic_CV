# CV-Intelligence

Système de tri automatisé de CVs par IA pour TechCore Liège (contexte fictif).  
Conformité RGPD et EU AI Act (haut risque, Annexe III).

---

## État d'avancement

| Phase | Objectif | Statut |
|---|---|---|
| 1 | Infrastructure n8n + API FastAPI | ✅ Terminé |
| 2 | Parsing, anonymisation, pseudo-labels | ✅ Terminé |
| 3 | Entraînement du modèle ML | ✅ Terminé |
| 4 | Audit biais + Dashboard RH + conformité | 📅 Planifié |
| 5 | Tests, client pilote, documentation finale | 📅 Planifié |

---

## Structure du projet

```
cv-intelligence/
├── pipeline_ml/
│   ├── parse_cv.py          # Parsing + pseudonymisation des CV .txt
│   ├── pseudo_labels.py     # Génération des labels par scoring heuristique
│   └── train.py             # Entraînement et évaluation du modèle ML
│
├── data/
│   ├── raw/                 # CV bruts .txt (ignoré par Git — RGPD)
│   └── processed/           # features.csv + identities.csv (ignoré par Git)
│
├── models/                  # model.pkl, scaler.pkl (ignoré par Git)
├── reports/                 # evaluation.txt (ignoré par Git)
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
- Packages : `scikit-learn`, `pandas`, `joblib`, `xgboost`

```bash
pip install scikit-learn pandas joblib xgboost
```

---

## Utilisation — Pipeline ML (Phase 2 & 3)

### 1. Placer les CV dans `data/raw/`

Fichiers `.txt` structurés avec les sections suivantes :

```
Name: Jean Dupont
Gender: Male
Date of Birth: 1990-01-15
Address: 12 rue de la Paix, Liège, Belgium
Email: jean.dupont@email.com
Phone: +32-...
Target Role: Software Engineer

Education:
Master of Science — Computer Science — ULiège — 2015

Experience:
Software Engineer — TechCorp — Liège — 2015-09 to 2020-06
...

Skills:
Technical: Python, SQL, Docker, Git
Methods: Agile, CI/CD
Management: Mentoring

Languages:
English — C1
French — Native

Certifications:
AWS Certified Developer
```

### 2. Parser les CV

```bash
python pipeline_ml/parse_cv.py
```

Produit deux fichiers dans `data/processed/` :

| Fichier | Contenu |
|---|---|
| `features.csv` | 16 features anonymisées par CV + colonne `label` vide |
| `identities.csv` | Données sensibles (nom, email, téléphone, genre, âge) liées par `cv_id` |

> `identities.csv` ne doit **jamais** être passé au modèle ML.

### 3. Générer les pseudo-labels

```bash
python pipeline_ml/pseudo_labels.py
```

Remplit la colonne `label` dans `features.csv` par scoring heuristique (seuil 12/16 points).  
Les CV avec un label déjà renseigné (vrais labels recruteur) sont conservés tels quels.

### 4. Entraîner le modèle

```bash
python pipeline_ml/train.py
```

Entraîne et compare Régression Logistique, Random Forest et XGBoost.  
Sauvegarde le meilleur modèle dans `models/`.

Résultats actuels (205 CV synthétiques, pseudo-labels) :

| Modèle | F1 (test) | ROC-AUC |
|---|---|---|
| Régression Logistique | 0.810 | 0.948 |
| **Random Forest** | **0.837** | 0.924 |
| XGBoost | 0.837 | 0.943 |

### 5. Scorer un nouveau CV

```python
import joblib
import numpy as np

model        = joblib.load("models/model.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

# features_dict = résultat de parse_cv() sur un CV
X = np.array([[features_dict[col] for col in feature_cols]])
score = model.predict_proba(X)[0][1]  # probabilité d'invitation (0.0 – 1.0)
```

---

## Architecture RGPD

```
CV brut
   │
   ├─ parse_cv.py
   │     ├─ identities.csv  ←  nom, email, téléphone, genre, âge  (jamais au modèle)
   │     └─  features.csv   ←  16 features anonymisées             (input ML)
   │
   ├─ pseudo_labels.py  →  features.csv avec colonne label
   │
   └─ train.py          →  models/model.pkl
```

Chaque candidat reçoit un `cv_id` UUID déterministe à l'ingestion.  
Les deux fichiers se joignent uniquement via `cv_id`, jamais via des données nominatives.

---

## Conformité EU AI Act

Le recrutement est classé **usage à haut risque** (Annexe III). Obligations appliquées :

- Décision finale toujours humaine — le modèle produit un score, pas une décision
- Aucun rejet définitif sans révision humaine tracée
- Données sensibles (`gender`, `age`) exclues du modèle — monitoring biais uniquement
- Explicabilité SHAP prévue en Phase 4
- Audit biais (Fairlearn) prévu en Phase 4

---

## Infrastructure Phase 1 (n8n)

Le pipeline d'ingestion automatique est opérationnel sur `https://n8n.lony.app`.  
Voir `docs/checkup/n8n.md` pour le détail de l'architecture.

---

## Documentation

| Fichier | Contenu |
|---|---|
| `docs/roadmap.md` | Roadmap complète du projet |
| `docs/entreprise_fictive.md` | Contexte TechCore + logique du bootstrapping |
| `docs/checkup/n8n.md` | Architecture pipeline n8n + API |
| `docs/checkup/model.md` | Résultats ML, interprétation, limites, prochaines étapes |
