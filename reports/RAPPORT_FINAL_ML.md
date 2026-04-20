# Rapport de Projet : CV Screening

**Candidats :** Tom Perez Le Tiec | Arnaud Leroy  
**Date :** 20 Avril 2026

---

## Contexte & Objectif

CV-Intelligence est un système de **pré-filtrage automatique** de candidatures — l'équivalent d'un filtre anti-spam avant l'intervention du RH. À l'échelle d'une grande entreprise (250 000 candidatures/mois chez Google), un tri manuel est impossible. L'objectif n'est pas de classer finement les candidats, mais d'éliminer les candidatures hors-sujet pour que le RH ne traite que les profils pertinents.

**Conséquence directe sur les métriques :** on optimise le **recall** (ne pas rater un bon candidat) plus que la précision (ne pas sur-inviter). Un faux négatif est une perte définitive ; un faux positif est géré par le RH en quelques secondes.

---

## Architecture Technique

### Pipeline (6 étapes)

```
p01_parse.py      → Parsing des 500 CVs bruts (.txt) → features.csv + identities.csv
p02_features.py   → Feature engineering v3 (36 colonnes)
p03_analysis.py   → EDA : outliers, VIF, mutual information
p04_train.py      → Entraînement + seuils différenciés → model.pkl
p06_audit.py      → Audit équité (genre, âge, pays) + SHAP
```

### Modèle

Régression Logistique avec `C=0.1` (régularisation forte) et `class_weight='balanced'` (dataset déséquilibré : 80% rejetés / 20% invités). Simple, stable, probabilités directement utilisables.

---

## Feature Engineering v3

### Variables du modèle (9 variables)

| Variable | Description | SHAP | Pourquoi |
|---|---|---|---|
| `education_level` | Niveau de diplôme (1-4) | 0.529 | Critère de sélection le plus stable |
| `career_depth` | Expérience × Durée moyenne | 0.288 | Profondeur de carrière |
| `potential_score` | (Skills + Méthodes + Certif) / (Exp+1) | 0.268 | Valorise les profils à fort potentiel |
| `has_multiple_languages` | 1 si ≥ 2 langues | 0.203 | Signal de profil international |
| `is_it` | 1 si secteur Informatique | 0.151 | Secteur dominant dans le dataset |
| `field_match` | Formation cohérente avec le secteur visé | 0.116 | Pertinence de la candidature |
| `junior_potential` | `is_junior × potential_score` | 0.088 | Booste les juniors à fort potentiel |
| `exp_per_year_of_age` | `years_experience / max(age-22, 1)` | 0.057 | Exp. normalisée par durée de carrière possible |
| `avg_job_duration` | Durée moyenne par poste | 0.034 | Stabilité de carrière |

### Évolution v5 → v3

La principale évolution est le **remplacement de `years_experience` par `exp_per_year_of_age`**.

| Problème v5 | Solution v3 |
|---|---|
| `years_experience` : SHAP #1 (0.52), structurellement défavorable aux femmes (pauses carrière) et aux juniors | `exp_per_year_of_age = years_experience / max(age-22, 1)` normalise par la durée de carrière *possible* |
| `is_finance` : colinéaire avec `is_it` | Supprimée |
| — | `field_match` : adéquation formation / secteur (Informatique, Finance, Industrie) |

**Résultat :** `exp_per_year_of_age` tombe au rang #8 (SHAP 0.057). Le modèle ne dépend plus d'une feature intrinsèquement biaisée comme signal principal.

---

## Correction d'Équité : Seuils Différenciés par Âge

Le modèle applique deux seuils de décision pour ne pas pénaliser les jeunes candidats, qui ont mécaniquement moins d'années de carrière.

| Groupe | Seuil appliqué | Objectif |
|---|---|---|
| Adultes (30+) | **0.462** | Maximise le F1-score |
| Juniors (<30) | **0.494** | Recall ≥ 0.55 avec meilleure précision possible |

---

## Métriques de Performance (v3)

### Sur le dataset complet (500 CV labellisés)

| Métrique | v5 (baseline) | **v3** | Δ |
|---|---|---|---|
| **ROC-AUC** | 0.785 | **0.797** | +0.012 |
| Recall global | — | **0.85** | — |
| Accuracy | 0.73 | 0.70 | -0.03 |
| F1-Score (Invité) | 0.49 | **0.53** | +0.04 |

> L'accuracy baisse légèrement car le modèle invite davantage (cohérent avec l'objectif recall élevé d'un pré-filtrage anti-spam).

### Matrice de Confusion (500 CV)

|  | Prédit Rejeté | Prédit Invité |
|---|---|---|
| **Réel Rejeté** | ~265 (VN) | ~135 (FP) |
| **Réel Invité** | ~15 (FN) | ~85 (VP) |

![Matrice de Confusion](plots/confusion_matrix.png)

### Courbe ROC

![Courbe ROC](plots/roc_curve.png)

---

## Importance des Variables (SHAP)

| Rang | Variable | Impact SHAP |
|---|---|---|
| 1 | `education_level` | 0.529 |
| 2 | `career_depth` | 0.288 |
| 3 | `potential_score` | 0.268 |
| 4 | `has_multiple_languages` | 0.203 |
| 5 | `is_it` | 0.151 |
| 6 | `field_match` | 0.116 |
| 7 | `junior_potential` | 0.088 |
| 8 | `exp_per_year_of_age` | 0.057 |
| 9 | `avg_job_duration` | 0.034 |

![Importance des Variables](plots/feature_importance.png)

---

## Audit d'Équité (v3)

### Par Genre

| Groupe | n | Recall | Précision |
|---|---|---|---|
| Femmes | 233 | **0.773** | 0.370 |
| Hommes | 267 | **0.786** | 0.400 |
| **Écart** | — | **+1.3 pts** | — |

**Amélioration majeure :** l'écart de recall genre passe de **13 pts (v5) à 1.3 pt (v3)** grâce au remplacement de `years_experience` par `exp_per_year_of_age`. Aucun attribut protégé (genre, âge) n'est utilisé comme feature du modèle.

![Équité](plots/fairness_metrics.png)

### Par Âge

| Groupe | n | Recall |
|---|---|---|
| Adulte (30-45) | 321 | 0.829 |
| Jeune (<30) | 179 | 0.556 |

L'écart résiduel juniors/adultes est structurel : les juniors ont moins de signal (peu d'expérience, peu de certifications). Le seuil abaissé compense partiellement.

**Alerte :** Aucun profil Senior (>45) dans le dataset — le modèle n'a pas été entraîné sur ce segment.

### Par Pays

| Pays | n | Recall |
|---|---|---|
| Portugal | 44 | 0.900 |
| Allemagne | 45 | 0.917 |
| Inde | 51 | 0.833 |
| Pologne | 57 | 0.857 |
| USA/Canada | 49 | 0.800 |
| Irlande | 50 | 0.727 |
| France | 50 | 0.700 |
| Italie | 48 | 0.714 |
| Nigeria | 61 | 0.667 |
| Pays-Bas | 45 | 0.667 |

Le modèle ne voit pas le pays directement. L'écart entre pays (Allemagne 0.917 vs Pays-Bas 0.667) reflète les distributions différentes des features selon les origines géographiques, pas un biais direct.

![Équité par Pays](plots/fairness_country.png)

---

## Conclusion

Le modèle v3 (AUC=0.797, 500 CV) remplit son rôle de **pré-filtrage anti-spam** : recall élevé, biais genre quasi-éliminé, pipeline propre en 6 étapes.

### Apports v3 vs v5

| Axe | v5 | v3 |
|---|---|---|
| AUC | 0.785 | **0.797** |
| Écart genre | 13 pts | **1.3 pts** |
| Feature principale | `years_experience` (biaisée) | `education_level` (neutre) |
| Pipeline | 12 scripts | **6 scripts** |

### Limites restantes

- Absence de profils Senior (>45) dans les données d'entraînement
- Écart résiduel juniors/adultes (recall 0.556 vs 0.829) — structurel, lié au manque de signal
- Précision "Invité" modeste (0.39) : coût assumé du pré-filtrage à haute sensibilité
- Dataset synthétique : `cv_completeness` et `red_flag_count` (features anti-spam réelles) non-pertinentes ici
