# Rapport de Projet : CV Screening

**Candidats :** Tom Perez Le Tiec | Arnaud Leroy  
**Date :** 20 Avril 2026

---

## Architecture Technique

### Modèle Utilisé
On utilise une Régression Logistique avec une régularisation assez forte (C=0.1). C'est simple, stable, et ça donne des probabilités directement utilisables pour classer les candidats. Le dataset est déséquilibré (80% rejetés / 20% invités), donc on a mis `class_weight='balanced'` pour que le modèle ne ignore pas les profils invités.

### Variables Utilisées (8 variables)

| Variable | Description | SHAP | Pourquoi |
|---|---|---|---|
| `years_experience` | Années d'expérience totales | 0.523 | Variable la plus importante |
| `education_level` | Niveau d'études (1-4) | 0.440 | Stable, peu d'outliers |
| `has_multiple_languages` | 1 si ≥ 2 langues | 0.182 | Remplace `nb_languages` (variable quasi-binaire dans ce dataset) |
| `career_depth` | Expérience × Durée moyenne | 0.133 | Profondeur de carrière |
| `potential_score` | (Compétences + Méthodes + Certif) / (Exp+1) | 0.096 | Valorise les profils à fort potentiel |
| `is_it` | 1 si secteur Informatique | 0.062 | Secteur dominant dans le dataset |
| `avg_job_duration` | Durée moyenne par poste | 0.049 | Stabilité de carrière |
| `junior_potential` | `is_junior × potential_score` | 0.045 | Booste les juniors à fort potentiel |

> **Note :** `is_finance` supprimée (colinéarité avec `is_it`). `nb_languages` remplacé par `has_multiple_languages` (majorité des CV ont 1 ou 2 langues). `junior_potential` est un terme d'interaction (`years_experience < 3` × `potential_score`) pour valoriser différemment le potentiel selon le stade de carrière.

---

## Feature Engineering

- **`potential_score`** = `(Compétences + Méthodes + Certifications) / (Expérience+1)` : valorise quelqu'un qui progresse vite même avec peu d'expérience.
- **`junior_potential`** : Terme d'interaction qui donne un signal supplémentaire aux juniors à fort potentiel.
- **Suppression de `is_finance`** : Colinéarité avec `is_it`.
- **`nb_languages` → `has_multiple_languages`** : La variable binaire est plus robuste et plus simple.
- **Winsorisation (percentile 5-95)** : Appliquée sur plusieurs variables pour limiter l'influence des valeurs extrêmes.

---

## Correction d'Équité : Seuils Différenciés par Âge

Le modèle utilise deux seuils de décision distincts pour ne pas pénaliser les jeunes candidats. `years_experience` (variable #1 en importance SHAP) est structurellement défavorable aux moins de 30 ans, qui ont mécaniquement moins d'années de carrière.

| Groupe | Seuil appliqué | Recall avant correction | Recall après correction |
|---|---|---|---|
| Adultes (30+) | **0.614** | 0.88 | 0.66 |
| Juniors (<30) | **0.374** | 0.26 | 0.56 |

Le seuil junior est abaissé à 0.374 (vs 0.614) : le modèle est délibérément plus permissif avec les jeunes candidats pour ne pas rater des talents à fort potentiel.

**Trade-off assumé :** Le taux d'invitation global passe à 31.6% (vs 20% dans les labels). La précision sur "Invité" baisse en contrepartie — c'est le coût de la non-discrimination.

---

## Comparaison : Modèle IA vs Labels

| Source | Dataset | Taux d'Invitation | Écart |
|:---|:---|:---|:---|
| Labels | 500 CV | 20.0% | Référence |
| Modèle IA (Fairness-Aware) | 500 CV | 31.6% | +11.6 pp |

L'écart s'explique principalement par la correction d'équité sur les juniors : le modèle invite davantage dans cette population pour compenser le biais structurel de `years_experience`.

---

## Métriques de Performance

### Sur le dataset complet (500 CV labellisés)

| Métrique | Valeur |
|---|---|
| **ROC-AUC** | **0.785** |
| Seuil adultes (30+) | 0.614 |
| Seuil juniors (<30) | 0.374 |
| Accuracy | 0.73 |
| F1-Score (Invité) | 0.49 |

### Matrice de Confusion (500 CV)

|  | Prédit Rejeté | Prédit Invité |
|---|---|---|
| **Réel Rejeté** | 306 (VN) | 94 (FP) |
| **Réel Invité** | 36 (FN) | 64 (VP) |

![Matrice de Confusion](plots/confusion_matrix.png)

### Courbe ROC

![Courbe ROC](plots/roc_curve.png)

---

## Importance des Variables (SHAP)

| Rang | Variable | Impact SHAP |
|---|---|---|
| 1 | `years_experience` | 0.523 |
| 2 | `education_level` | 0.440 |
| 3 | `has_multiple_languages` | 0.182 |
| 4 | `career_depth` | 0.133 |
| 5 | `potential_score` | 0.096 |
| 6 | `is_it` | 0.062 |
| 7 | `avg_job_duration` | 0.049 |
| 8 | `junior_potential` | 0.045 |

![Importance des Variables](plots/feature_importance.png)

---

## Audit d'Équité

On a fusionné `features.csv` avec `identities.csv` pour vérifier les biais sur 500 CV.

### Par Genre

| Groupe | n | Recall | Précision |
|---|---|---|---|
| Femmes | 233 | 0.57 | 0.34 |
| Hommes | 267 | 0.70 | 0.46 |

Légère disparité en faveur des hommes (+13% recall). À surveiller sur un dataset plus grand.

### Par Âge (après correction)

| Groupe | n | Taux invitation labels | Recall |
|---|---|---|---|
| Adulte (30+) | 321 | 25.5% | 0.66 |
| Jeune (<30) | 179 | 10.1% | 0.56 |

Les juniors sont nettement moins souvent invités dans les labels (10% vs 25%). La correction d'équité ramène leur recall à 0.56, proche des adultes.

**Alerte :** Pas de profils Senior (>45) dans le dataset — le modèle n'a pas été entraîné sur ce segment.

![Équité](plots/fairness_metrics.png)

### Par Pays

| Pays | n | Recall | Précision |
|---|---|---|---|
| Pays-Bas | 45 | 0.833 | 0.357 |
| Allemagne | 45 | 0.750 | 0.562 |
| Inde | 51 | 0.750 | 0.474 |
| Portugal | 44 | 0.700 | 0.636 |
| Pologne | 57 | 0.714 | 0.385 |
| France | 50 | 0.600 | 0.316 |
| Nigeria | 61 | 0.600 | 0.409 |
| Irlande | 50 | 0.545 | 0.353 |
| USA/Canada | 49 | 0.500 | 0.312 |
| Italie | 48 | 0.429 | 0.273 |

Le modèle ne voit pas le pays directement — les écarts viennent des distributions différentes des variables selon les origines géographiques.

![Équité par Pays](plots/fairness_country.png)

---

## Conclusion

Le modèle (AUC=0.785, 500 CV) fonctionne bien pour trier des CV par ordre de priorité.

La principale nouveauté de cette version est la **correction d'équité par seuil différencié** : le modèle utilise un seuil abaissé (0.374) pour les candidats de moins de 30 ans, ce qui corrige le biais structurel de `years_experience` et ramène leur recall de 0.26 à 0.56.

Limites restantes :
- Absence de profils Senior (>45) dans les données d'entraînement
- Légère disparité de recall Hommes/Femmes (+13% en faveur des hommes)
- Précision "Invité" modeste (0.39) : le modèle sur-invite pour ne pas rater de candidats
