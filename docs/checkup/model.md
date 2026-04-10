# Checkup — Modèle ML (Phase 3)

*Dernière mise à jour : Avril 2026*

---

## 1. Ce qu'on a entraîné

### Données d'entrée

- **Source** : `data/processed/features.csv` — 205 CV synthétiques (TechCore, contexte fictif)
- **Labels** : pseudo-labels générés par scoring heuristique (`pipeline_ml/pseudo_labels.py`)
- **Distribution** : 103 invités (1) / 102 rejetés (0) — parfaitement équilibré (50/50)
- **Valeurs manquantes** : 0
- **Données sensibles dans le modèle** : aucune — `age`, `gender`, `nom`, `email`, `téléphone` sont dans `identities.csv` uniquement

### Features utilisées (16 colonnes)

| Feature | Type | Description |
|---|---|---|
| `education_level` | Ordinal (1–4) | 1=sans diplôme, 2=Bachelor, 3=Master, 4=PhD |
| `nb_jobs` | Entier | Nombre de postes |
| `years_experience` | Float | Années d'expérience cumulées |
| `avg_job_duration` | Float | Durée moyenne par poste |
| `career_progression` | Binaire | Progression détectée (senior + changement d'entreprise) |
| `nb_technical_skills` | Entier | Nombre de compétences techniques |
| `nb_methods_skills` | Entier | Nombre de méthodes |
| `nb_management_skills` | Entier | Nombre de compétences management |
| `total_skills` | Entier | Somme des 3 précédents |
| `nb_languages` | Entier | Nombre de langues |
| `has_english` | Binaire | Anglais présent |
| `english_level` | Ordinal (0–6) | Niveau CECRL (A1=1 … C2=6) |
| `has_french` | Binaire | Français présent |
| `has_german` | Binaire | Allemand présent |
| `has_luxembourgish` | Binaire | Luxembourgeois présent |
| `nb_certifications` | Entier | Nombre de certifications |

### Split

| Set | Taille | % |
|---|---|---|
| Train | 131 | 64% |
| Validation | 33 | 16% |
| Test | 41 | 20% |

Split stratifié (`stratify=y`) — proportions 50/50 conservées dans chaque set.

---

## 2. Résultats

### Cross-validation sur train (5-fold, F1)

| Modèle | F1 moyen | Écart-type |
|---|---|---|
| Régression Logistique | 0.820 | ± 0.029 |
| Random Forest | 0.878 | ± 0.065 |
| XGBoost | 0.875 | ± 0.092 |

### Test set (données jamais vues)

| Modèle | F1 | ROC-AUC |
|---|---|---|
| Régression Logistique | 0.810 | **0.948** |
| **Random Forest** | **0.837** | 0.924 |
| XGBoost | 0.837 | 0.943 |

**Modèle retenu : Random Forest** (meilleur F1 test, variance CV correcte).

### Confusion matrix — Random Forest (test set, 41 CV)

```
                 Prédit Rejeté   Prédit Invité
Réel Rejeté          16               4
Réel Invité           3              18
```

- 4 faux positifs (rejetés classés invités) — cas où le modèle est trop optimiste
- 3 faux négatifs (invités classés rejetés) — cas où le modèle rate un bon profil

Dans un contexte recrutement, **les faux négatifs sont plus coûteux** (on rate un bon candidat). À surveiller en production.

---

## 3. Interprétation honnête des résultats

### Ce qui explique les bons scores

Le modèle apprend à reproduire les règles du scoring heuristique, pas à prédire le comportement réel d'un recruteur. C'est attendu et voulu à ce stade — c'est le principe du bootstrapping.

La Régression Logistique obtient un ROC-AUC de **0.948**, ce qui indique que les features sont **linéairement bien séparables**. C'est logique : les pseudo-labels ont été calculés avec une somme pondérée des mêmes features. Le modèle retrouve essentiellement cette fonction de scoring.

### Ce que ces scores ne garantissent pas

- **Pas de généralisation sur de vrais CV** : les CVs sont synthétiques et uniformément bien structurés. En production, les CVs seront bruités, mal formatés, avec des champs manquants.
- **Pas de validité prédictive réelle** : un F1 de 0.837 mesure la capacité à reproduire les pseudo-labels, pas à prédire les décisions réelles d'un recruteur.
- **Overfitting potentiel** : Random Forest avec 200 arbres sur 131 exemples — le modèle est surdimensionné pour la taille du dataset. En production avec peu de données, préférer la Régression Logistique (plus robuste, variance plus faible).

### Ce qu'on attend lors du passage aux vrais labels

Quand les recruteurs valideront/rejetteront de vrais candidats :
1. Les scores F1 vont **baisser** — les vrais labels sont plus bruités que les pseudo-labels
2. Puis **remonter progressivement** au fil des retours — c'est le signe que le modèle apprend quelque chose de réel
3. L'écart entre pseudo-labels et vrais labels indiquera quelles règles métier étaient fausses ou incomplètes

---

## 4. Fichiers produits

| Fichier | Contenu |
|---|---|
| `models/model.pkl` | Random Forest entraîné (joblib) |
| `models/scaler.pkl` | StandardScaler fitté sur train |
| `models/feature_cols.pkl` | Liste ordonnée des 16 features |
| `reports/evaluation.txt` | Métriques du test set |

Pour scorer un nouveau CV :
```python
import joblib, numpy as np

model       = joblib.load("models/model.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

# features_dict = résultat de parse_cv() sur un nouveau CV
X = np.array([[features_dict[col] for col in feature_cols]])
score = model.predict_proba(X)[0][1]  # probabilité d'invitation (0.0–1.0)
```

---

## 5. Prochaines étapes

### Phase 4 — Audit biais (prioritaire avant production)

Croiser les prédictions du modèle avec `identities.csv` (genre, âge) pour calculer :
- **Disparate Impact Ratio** par genre : taux d'invitation femmes / taux d'invitation hommes (objectif >= 0.80, cible > 0.95)
- **Demographic Parity Difference** via Fairlearn (objectif proche de 0)

Le modèle n'utilise pas `gender` comme feature, mais il peut quand même être biaisé indirectement si certaines features corrèlent avec le genre dans les données.

### Phase 4 — Explicabilité SHAP

Calculer les SHAP values pour chaque prédiction afin d'expliquer au recruteur quelles features ont le plus contribué au score. Obligation d'explicabilité AI Act.

### Collecte de vrais labels

Dès les premières candidatures réelles, logger les décisions des recruteurs et les associer aux `cv_id`. Ces vrais labels remplaceront progressivement les pseudo-labels pour réentraîner le modèle.
