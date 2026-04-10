# Roadmap — CV-Intelligence

## État d'avancement

| Phase | Objectif | Statut |
|-------|----------|--------|
| 1 | Infrastructure & automatisation n8n | ✅ Terminé |
| 2 | Pré-traitement des données (parsing, anonymisation, feature engineering) | ✅ Terminé |
| 3 | Entraînement du modèle ML + audit biais + SHAP | ✅ Terminé |
| 4 | Dashboard RH + conformité + intégration n8n → ML | 🔄 En cours |
| 5 | Tests, données supplémentaires, client pilote, documentation finale | 📅 Planifié |

---

## Phase 1 — Infrastructure & Automatisation n8n ✅

**Terminé.** Le pipeline d'ingestion automatique est opérationnel.

- Serveur Ubuntu (DigitalOcean) + Nginx reverse proxy + SSL Let's Encrypt
- Instance n8n sur `https://n8n.lony.app`
- Workflow n8n en 5 nœuds : Schedule → GET emails → Filter (pièces jointes) → Download binary → POST multipart vers FastAPI
- API FastAPI `cv_api` reçoit le CV et renvoie un `candidate_id` avec statut `awaiting_review`
- Communication interne Docker via `http://api:8000`

---

## Phase 2 — Pré-traitement des données ✅

**Terminé.** Pipeline de parsing, pseudonymisation et feature engineering opérationnel.

### Ce qui a été fait

**Parsing** (`pipeline_ml/parse_cv.py`)
- 205 CVs `.txt` parsés en parallèle (ThreadPoolExecutor)
- Séparation RGPD stricte : `features.csv` (ML) + `identities.csv` (données sensibles)
- `cv_id` UUID déterministe par candidat
- Colonne `profile_type` (junior / intermediate / senior) pour filtre dashboard
- Labels chargés depuis `pipeline_ml/student_labels.csv` (200 labels réels)

**Feature engineering** (`pipeline_ml/feature_engineering.py`)
- 9 features composites ajoutées : `log_years_exp`, `exp_edu_score`, `cert_density`, `multilingual_score`, `method_tech_ratio`, `tech_per_year`, `career_depth`, `is_it`, `is_finance`
- `exp_edu_score` (r=+0.364) meilleure corrélation avec le label
- `tech_per_year` (r=-0.243) détecte les CV gonflés en skills

**Note** : les hard filters ont été abandonnés — avec un modèle ML, les rejets sont gérés par le score + révision humaine, conformément à l'AI Act.

---

## Phase 3 — Entraînement du modèle ML ✅

**Terminé.** Modèle entraîné sur 200 CVs avec labels réels.

### Résultats (Avril 2026)

| Modèle | F1 (test) | ROC-AUC |
|---|---|---|
| **Régression Logistique** | **0.621** | **0.837** |
| Random Forest | 0.267 | 0.727 |
| XGBoost | 0.333 | 0.640 |

- Dataset : 200 CVs labellisés (51 invités / 149 rejetés, ratio 25/75)
- 5 CVs sans label ignorés (`cv1` à `cv5`, hors `student_labels.csv`)
- Modèle retenu : **Régression Logistique** (meilleur F1 + AUC, plus stable sur petit dataset)
- Scores limités par la taille du dataset — amélioration attendue avec données supplémentaires

### Audit biais

- Parité genre : DI = 0.992 ✅
- Parité âge : DI = 0.312 — structurel (juniors ont moins d'expérience), Equal Opportunity OK
- SHAP global disponible dans `reports/audit.txt`
- Voir `docs/checkup/model_audit.md` pour l'analyse complète

---

## Phase 4 — Dashboard RH + Conformité 🔄

En cours. Frontend développé par un autre groupe.

**Côté pipeline (à faire) :**
- Intégration n8n → endpoint de scoring FastAPI (Nœud 6)
- Calibration des probabilités (Platt Scaling)
- Encodage sémantique de `target_role`
- Ré-entraînement automatique dès 50 nouveaux vrais labels

**Dashboard (autre groupe) :**
- Classement par score avec filtre `profile_type` (junior / intermediate / senior)
- Explications SHAP par candidat
- Actions manuelles recruteur (override, motif)
- Rapport biais exportable

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
