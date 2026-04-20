# Rapport de Projet : CV Screening

**Candidats :** Tom Perez Le Tiec | Arnaud Leroy  
**Date :** 14 Avril 2026

---

## Architecture Technique

### Modèle Utilisé
On utilise une Régression Logistique avec une régularisation assez forte (C=0.1). C'est simple, stable, et ça donne des probabilités directement utilisables pour classer les candidats. Le dataset est déséquilibré (74.5% rejetés / 25.5% invités), donc on a mis `class_weight='balanced'` pour que le modèle ne ignore pas les profils invités.

### Variables Utilisées (7 variables)

| Variable | Description | SHAP | Pourquoi |
|---|---|---|---|
| `years_experience` | Années d'expérience totales | 0.442 | Variable la plus importante |
| `education_level` | Niveau d'études (1-4) | 0.359 | Stable, peu d'outliers |
| `has_multiple_languages` | 1 si ≥ 2 langues | 0.300 | Remplace `nb_languages` (variable quasi-binaire dans ce dataset) |
| `potential_score` | (Compétences + Méthodes + Certif) / (Exp+1) | 0.151 | Valorise les profils à fort potentiel |
| `avg_job_duration` | Durée moyenne par poste | 0.088 | Stabilité de carrière |
| `junior_potential` | `is_junior × potential_score` | 0.076 | Booste les juniors à fort potentiel |
| `is_it` | 1 si secteur Informatique | 0.031 | 78.5% du dataset |
| `career_depth` | Expérience × Durée moyenne | 0.002 | Profondeur de carrière |

> **Note :** `is_finance` supprimée (colinéarité parfaite avec `is_it`). `nb_languages` remplacé par `has_multiple_languages` (85% des CV ont exactement 2 langues). `junior_potential` est un terme d'interaction (`years_experience < 3` × `potential_score`) pour valoriser différemment le potentiel selon le stade de carrière.

---

## Feature Engineering

- **Suppression de `exp_edu_score`** : Ce score (expérience × diplôme) pénalisait massivement les jeunes. On l'a enlevé.
- **`potential_score`** = `(Compétences + Méthodes + Certifications) / (Expérience+1)` : ça permet de valoriser quelqu'un qui progresse vite même avec peu d'expérience.
- **`junior_potential`** : Terme d'interaction qui donne un signal supplémentaire aux juniors à fort potentiel. Sans ça, le modèle applique le même poids à `potential_score` pour tout le monde.
- **Suppression de `is_finance`** : Colinéarité parfaite avec `is_it`.
- **Log-transform de `avg_job_duration`** : L'EDA montrait une skewness de 1.61 et 17.5% d'outliers. Le log stabilise la distribution.
- **`nb_languages` → `has_multiple_languages`** : 95% des CV ont 1 ou 2 langues. La variable binaire est plus robuste et plus simple.
- **Winsorisation (percentile 5-95)** : Appliquée sur plusieurs variables pour limiter l'influence des valeurs extrêmes.

---

## Comparaison : Modèle IA vs Labels

| Source             | Taux d'Invitation | Écart |
|:-------------------| :--- | :--- |
| Labels             | 25.5% | Référence |
| Modèle IA (v6)     | 30.0% | +4.5 pp |

Le modèle qu'on a fait invite un peu plus que le dataset de base (30% vs 25.5%). C'est plus ou moins voulu : on a choisi un seuil légèrement permissif (0.601) pour que les juniors ne soient pas tous éliminés à cause de `years_experience`.

---

## Métriques de Performance

### Sur le dataset complet (200 CV)

| Métrique | Valeur |
|---|---|
| **ROC-AUC** | **0.783** |
| Seuil de Tri Optimal | 0.601 |
| Accuracy | 0.765 |
| F1-Score (Invité) | 0.577 |

### Cross-Validation 5-Fold

| Métrique | Moyenne | Écart-type |
|---|---|---|
| **ROC-AUC** | **0.760** | ±0.021 |
| **F1-Score (Invité)** | **0.561** | ±0.059 |

> Un ROC-AUC de 0.780 signifie que le modèle classe correctement 78% des paires Invité/Rejeté. C'est stable par rapport à v5 (0.782) malgré la refonte EDA.

### Matrice de Confusion (200 CV)

|  | Prédit Rejeté | Prédit Invité |
|---|---|---|
| **Réel Rejeté** | 121 (VN) | 28 (FP) |
| **Réel Invité** | 19 (FN) | 32 (VP) |

![Matrice de Confusion](plots/confusion_matrix.png)

### Courbe ROC

![Courbe ROC](plots/roc_curve.png)

---

## Importance des Variables (SHAP)

| Rang | Variable | Impact SHAP |
|---|---|---|
| 1 | `years_experience` | 0.442 |
| 2 | `education_level` | 0.359 |
| 3 | `has_multiple_languages` | 0.300 |
| 4 | `potential_score` | 0.151 |
| 5 | `avg_job_duration` | 0.088 |
| 6 | `junior_potential` | 0.076 |
| 7 | `is_it` | 0.031 |
| 8 | `career_depth` | 0.002 |

![Importance des Variables](plots/feature_importance.png)

---

## Audit d'Équité

On a fusionné `features.csv` avec `identities.csv` pour vérifier les biais.

### Par Genre

| Groupe | n | Recall | Précision |
|---|---|---|---|
| Femmes | 89 | 0.727 | 0.593 |
| Hommes | 111 | 0.552 | 0.485 |

Le modèle favorise un peu les femmes (+17.5% recall). À surveiller sur un dataset plus grand.

### Par Âge

| Groupe | n | Recall | Précision |
|---|---|---|---|
| Adulte (30-45) | 132 | 0.714 | 0.536 |
| Jeune (<30) | 68 | 0.222 | 0.500 |

**Alerte :** Pas de profils Senior (>45) dans le dataset -> le modèle n'a pas été entraîné sur ce segment.

![Équité](plots/fairness_metrics.png)

### Par Pays

| Pays | n | Recall | Précision |
|---|---|---|---|
| Allemagne | 21 | 1.000 | 0.750 |
| Inde | 19 | 0.857 | 0.857 |
| Pays-Bas | 20 | 0.750 | 0.429 |
| Pologne | 20 | 1.000 | 0.250 |
| USA/Canada | 17 | 0.600 | 0.750 |
| Nigeria | 23 | 0.500 | 0.500 |
| Italie | 19 | 0.500 | 0.333 |
| Portugal | 17 | 0.500 | 0.667 |
| France | 18 | 0.429 | 0.600 |
| Irlande | 26 | 0.429 | 0.300 |

Le modèle ne voit pas le pays, les écarts viennent des distributions différentes des variables selon les origines.

![Équité par Pays](plots/fairness_country.png)

---

## Conclusion
Le modèle fonctionne bien pour ce qu'on lui demande : trier des CV par ordre de priorité.

Il reste des limites (pas de seniors dans les données, disparités par pays)
