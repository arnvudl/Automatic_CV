# Checkup — Modèle ML (Phase 3)

*Dernière mise à jour : Avril 2026*

---

## 1. Ce qu'on a entraîné

### Données d'entrée

- **Source** : `data/processed/features.csv` — 205 CVs parsés (200 avec labels réels)
- **Labels** : vrais labels recruteur depuis `pipeline_ml/student_labels.csv` (200 labels)
- **5 CVs sans label** (cv1–cv5, absents de student_labels.csv) — ignorés à l'entraînement
- **Distribution** : 51 invités (1) / 149 rejetés (0) — ratio 25/75 (déséquilibré, réel)
- **Données sensibles dans le modèle** : aucune — `age`, `gender`, `nom`, `email`, `téléphone` sont dans `identities.csv` uniquement

### Features utilisées (19 colonnes)

#### Features brutes

| Feature | Type | Description |
|---|---|---|
| `years_experience` | Float | Années d'expérience cumulées |
| `avg_job_duration` | Float | Durée moyenne par poste (stabilité) |
| `education_level` | Ordinal (1–4) | 1=sans diplôme, 2=Bachelor, 3=Master, 4=PhD |
| `nb_jobs` | Entier | Nombre de postes occupés |
| `nb_methods_skills` | Entier | Nombre de méthodes (Agile, Scrum…) |
| `nb_languages` | Entier | Nombre de langues parlées |
| `nb_certifications` | Entier | Nombre de certifications |
| `english_level` | Ordinal (0–6) | Niveau CECRL anglais (A1=1 … C2=6) |
| `has_german` | Binaire | Allemand présent |
| `nb_technical_skills` | Entier | Nombre de compétences techniques |

#### Features engineered (`pipeline_ml/feature_engineering.py`)

| Feature | Formule | Corrélation avec label |
|---|---|---|
| `log_years_exp` | log1p(years_experience) | atténue outliers seniors |
| `exp_edu_score` | years × education_level | **r=+0.364** — meilleur signal |
| `cert_density` | nb_certifications / nb_jobs | certifications par poste |
| `multilingual_score` | nb_languages + bonus anglais B2+ | signal langue composite |
| `method_tech_ratio` | nb_methods / nb_technical | équilibre profil |
| `tech_per_year` | nb_technical / years | **r=−0.243** — détecte CV gonflés |
| `career_depth` | years × avg_job_duration | **r=+0.257** — séniorité stable |
| `is_it` | secteur == "IT" | one-hot secteur |
| `is_finance` | secteur == "Finance" | one-hot secteur |

### Split stratifié

| Set | Taille | Invités | Rejetés |
|---|---|---|---|
| Train | ~128 | ~33 | ~95 |
| Validation | ~32 | ~8 | ~24 |
| Test | 40 | 10 | 30 |

Split 64/16/20 avec `stratify=y` — ratio 25/75 conservé dans chaque set.

---

## 2. Résultats (labels réels, Avril 2026)

### Test set (données jamais vues)

| Modèle | F1 | ROC-AUC |
|---|---|---|
| **Régression Logistique** | **0.621** | **0.837** |
| Random Forest | 0.267 | 0.727 |
| XGBoost | 0.333 | 0.640 |

**Modèle retenu : Régression Logistique** — meilleur F1 et AUC, plus stable sur petit dataset.

### Pourquoi pas Random Forest ou XGBoost ?

Random Forest et XGBoost sont trop complexes pour 200 exemples : ils sur-apprennent facilement les patterns du train set sans généraliser. La Régression Logistique est plus robuste avec peu de données, s'améliore linéairement à mesure que les données augmentent, et produit des probabilités calibrées — indispensable pour le score recruteur.

### Interprétation honnête

- F1=0.621 reflète la **vraie difficulté** du problème avec de vrais labels bruités (vs pseudo-labels qui donnaient F1=0.837)
- AUC=0.837 indique que le modèle **classe correctement** invités vs rejetés dans 84% des paires
- Les scores progresseront avec plus de données — amélioration attendue dès 300–400 CVs labellisés

---

## 3. Gestion du déséquilibre de classes

25% invités / 75% rejetés → sans correction, le modèle prédirait "rejeté" par défaut.

Solution : `class_weight='balanced'` dans la Régression Logistique — pénalise davantage les erreurs sur la classe minoritaire (invités).

---

## 4. Conformité RGPD & AI Act

- `gender` et `age` **absents** des features du modèle — monitoring biais uniquement
- `heuristic_score` conservé dans `features.csv` : score avant seuil pour contestation humaine
- Chaque décision finale reste humaine — le modèle produit un score, jamais une décision
- SHAP disponible via `pipeline_ml/audit.py` pour explicabilité au recruteur

---

## 5. Fichiers produits

| Fichier | Contenu |
|---|---|
| `models/model.pkl` | Régression Logistique entraînée (joblib) |
| `models/scaler.pkl` | StandardScaler fitté sur train |
| `models/feature_cols.pkl` | Liste ordonnée des 19 features |
| `reports/evaluation.txt` | Métriques du test set |
| `reports/audit.txt` | Audit biais + SHAP |

Pour scorer un nouveau CV :
```python
import joblib, numpy as np

model        = joblib.load("models/model.pkl")
scaler       = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

X = np.array([[features_dict[col] for col in feature_cols]])
score = model.predict_proba(scaler.transform(X))[0][1]  # 0.0–1.0
```

---

## 6. Prochaines étapes (Phase 4 & 5)

- **Calibration Platt Scaling** : améliorer la fiabilité des probabilités pour le dashboard
- **Encodage sémantique `target_role`** : feature texto actuellement ignorée
- **Ré-entraînement automatique** dès 50 nouveaux vrais labels
- **Données supplémentaires** en cours (professeur) — amélioration attendue avec 300+ CVs

---

*Pipeline complet : `python pipeline_ml/run.py full`*
