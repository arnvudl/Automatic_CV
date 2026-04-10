# Roadmap — CV-Intelligence

## État d'avancement

| Phase | Objectif | Statut |
|-------|----------|--------|
| 1 | Infrastructure & automatisation n8n | ✅ Terminé |
| 2 | Pré-traitement des données (parsing, anonymisation, hard filters) | 🔄 En cours |
| 3 | Entraînement du modèle ML | 📅 Planifié |
| 4 | Dashboard RH + audit biais + conformité | 📅 Planifié |
| 5 | Tests, client pilote, documentation finale | 📅 Planifié |

---

## Phase 1 — Infrastructure & Automatisation n8n ✅

**Terminé.** Le pipeline d'ingestion automatique est opérationnel.

- Serveur Ubuntu (DigitalOcean) + Nginx reverse proxy + SSL Let's Encrypt
- Instance n8n sur `https://n8n.lony.app`
- Workflow n8n en 5 nœuds : Schedule → GET emails → Filter (pièces jointes) → Download binary → POST multipart vers FastAPI
- API FastAPI `cv_api` reçoit le CV et renvoie un `candidate_id` avec statut `awaiting_review`
- Communication interne Docker via `http://api:8000`

---

## Phase 2 — Pré-traitement des données 🔄

C'est la phase en cours. L'objectif est de préparer les données pour entraîner le modèle ML, en respectant le cadre RGPD et AI Act.

Les CV bruts se trouvent dans `data/raw/`. Le parser de base (`pipeline_ml/extraction_cv.py`) est déjà fonctionnel.

### 2.1 Parsing (extraction des features)

Extraire les données structurées depuis les CV texte (`.txt`, `.pdf`).

- Utiliser `pipeline_ml/extraction_cv.py` comme base (Regex, rapide, sans coût de calcul)
- Features à extraire : `nb_jobs`, `years_experience`, `education_level`, `skills`, `languages`
- **Ne pas extraire** pour le modèle : `gender`, `age`, `nom`, `prénom`, `nationalité` — données sensibles exclues du ML (voir §RGPD)
- Output : fichier CSV structuré `data/processed/features.csv`

### 2.2 Anonymisation / Pseudonymisation

Le modèle ML ne doit jamais voir les données d'identité.

- Générer un `cv_id` (UUID) par candidat dès l'ingestion
- Séparer en deux flux :
  - **Identités** (`cv_id`, `nom`, `email`, `téléphone`) → stockées séparément, jamais envoyées au modèle
  - **Features** (`cv_id`, features extraites) → seules données vues par le ML
- Les données sensibles (`gender`, `age`) sont stockées uniquement à des fins de monitoring biais, jamais comme input du modèle

### 2.3 Hard Filters (critères minimums)

Filtre déterministe appliqué **avant** le ML, pour rejeter les profils manifestement hors critères sans solliciter le modèle.

Exemples de règles :
- Expérience < 2 ans → non éligible
- Aucune compétence requise présente → non éligible
- CV illisible / parsing échoué → revue manuelle

**Important (RGPD)** : un rejet issu du hard filter est toujours soumis à révision humaine avant notification. Il ne s'agit pas d'un rejet automatique définitif.

---

## Phase 3 — Entraînement du modèle ML 📅

Une fois les données pré-traitées et validées.

- Entraîner un modèle (Régression Logistique ou Random Forest en premier, XGBoost ensuite)
- Entrée : features anonymisées du CSV `data/processed/features.csv`
- Sortie : score de compatibilité (probabilité, pas une décision binaire)
- Export : `model.pkl` (joblib) intégré dans le conteneur FastAPI
- Audit biais : vérifier le ratio d'acceptation par groupe (genre, âge) via Fairlearn

---

## Phase 4 — Dashboard RH + Conformité 📅

- Interface recruteur : classement par score, explications (SHAP), actions manuelles
- Le recruteur peut corriger ou rejeter toute suggestion du modèle
- Traçabilité : qui a revu quoi, quand, avec quel motif (log immuable)
- Rapport biais périodique exportable
- Voie de contestation candidat (demande de révision humaine)

---

## Conformité RGPD & AI Act

Le recrutement est classé **usage à haut risque** par l'AI Act (Annexe III). Les obligations majeures sont en vigueur en 2026.

### Ce que le système fait (et doit continuer à faire)

| Sujet | Règle appliquée |
|-------|----------------|
| Décision finale | Toujours humaine — le modèle produit des suggestions, jamais des décisions |
| Rejet | Aucun rejet définitif sans revue humaine tracée |
| Transparence | Les candidats sont informés de l'usage de l'IA et disposent d'une voie de contestation |
| Données utilisées | Minimisation stricte — seules les features pertinentes au poste sont analysées |
| Données sensibles | `gender`, `age` exclus du modèle — monitoring biais uniquement |
| Explicabilité | Le recruteur voit les critères de score et les raisons principales du classement |
| Audit biais | Vérification périodique du Disparate Impact Ratio par groupe |
| Traçabilité | Log de chaque action humaine (qui, quand, motif) |

### Ce qui est interdit

- Rejet automatique sans révision humaine réelle
- Score "caché" sans contrôle ni explication au recruteur
- Utilisation du genre, de l'âge ou de l'origine comme feature du modèle
- Collecte de données non pertinentes pour le recrutement

### Mise en place pratique

1. **Règle interne documentée** : aucun rejet définitif sans revue humaine
2. **Information candidat** : mention de l'IA dans l'avis de confidentialité et les emails de réponse
3. **Voie de contestation** : le candidat peut demander une révision humaine en cas de rejet
4. **Audits périodiques** : biais, faux négatifs, profils atypiques
5. **Validation d'un échantillon** : revue manuelle régulière de CV classés par le modèle, y compris les profils atypiques
6. **Documentation technique** : finalité du système, limites, données utilisées (exigence AI Act)

---

*Dernière mise à jour : Avril 2026*
