# Rapport de Projet : Système de Tri de CV Automatisé (v6)

**Candidats :** Tom Perez Le Tiec | Arnaud Leroy  
**Date :** 14 Avril 2026  
**Version du Modèle :** v6 (Potential & Fair Optimized — Log-Transformed Features)

---

## 1. Synthèse du Projet
L'objectif de ce projet est de concevoir un système de tri (ranking) de CV capable d'identifier les profils les plus pertinents pour un processus de recrutement. Contrairement à un système de décision automatique, ce modèle sert d'assistant au recruteur pour prioriser l'examen des candidatures.

L'éthique algorithmique a été placée au centre de la démarche. Nous avons fait le choix délibéré de privilégier la neutralité et la réduction des biais systémiques, même si cela a nécessité un compromis sur la performance brute du modèle.

---

## 2. Architecture Technique

### Modèle Utilisé
Le modèle retenu est une Régression Logistique avec une forte régularisation (C=0.1). Ce choix assure la stabilité des prédictions sur un dataset de 200 CV et permet de fournir des probabilités claires pour le classement. La gestion du déséquilibre des classes (ratio 74.5% / 25.5%, signalé par l'EDA) est assurée par le paramètre `class_weight='balanced'`.

### Variables Utilisées (Robust Features — 7 variables)
Nous avons sélectionné 7 variables pour leur stabilité et leur pertinence métier, extraites du dataset `features.csv` :

| Variable | Description | SHAP | Motivation |
|---|---|---|---|
| `years_experience` | Nombre total d'années d'expérience | 0.442 | Signal le plus fort |
| `education_level` | Niveau d'études (encodé 1-4) | 0.359 | Stable, peu d'outliers |
| `has_multiple_languages` | 1 si ≥ 2 langues, 0 sinon | 0.300 | Remplace `nb_languages` — variable quasi-binaire dans ce dataset |
| `potential_score` | (Compétences + Méthodes + Certif) / (Exp+1) | 0.151 | Valorise les profils à fort potentiel |
| `avg_job_duration` | Durée moyenne par poste | 0.088 | Stabilité de carrière |
| `junior_potential` | `is_junior × potential_score` | 0.076 | Interaction : booste les juniors à fort potentiel |
| `is_it` | 1 si secteur Informatique | 0.031 | 78.5% du dataset |
| `career_depth` | Expérience × Durée moyenne | 0.002 | Profondeur de carrière |

> **Note :** `is_finance` supprimée (colinéarité parfaite avec `is_it`). `nb_languages` remplacé par `has_multiple_languages` (85% des CV ont exactement 2 langues). `junior_potential` est un terme d'interaction (`years_experience < 3` × `potential_score`) permettant à la régression logistique de valoriser différemment le potentiel selon le stade de carrière.

---

## 3. Feature Engineering
L'étape d'ingénierie des variables a été le pivot de notre approche éthique et technique.

- **Suppression du score combiné (`exp_edu_score`)** : Initialement utilisé, ce score multipliait l'expérience par le diplôme, créant un biais massif contre les jeunes talents.
- **Introduction du Potential Score** : `potential_score = (Compétences + Méthodes + Certifications) / (Expérience+1)`. Valorise la vitesse d'apprentissage des profils juniors.
- **[v6] `junior_potential = is_junior × potential_score`** : Terme d'interaction permettant à la régression logistique de valoriser différemment le potential_score selon le stade de carrière. Sans cette feature, le modèle applique le même poids à `potential_score` pour tous les profils. Avec elle, un junior à fort potentiel reçoit un signal additionnel qui monte sa probabilité d'invitation.
- **Suppression de `is_finance`** : Colinéarité parfaite avec `is_it` (tous les CV sont IT ou Finance).
- **Log-transform de `avg_job_duration`** : L'exploration EDA a révélé une asymétrie de 1.61 et 17.5% d'outliers sur cette variable. Le log-transform stabilise la distribution avant l'entraînement de la régression logistique.
- **`nb_languages` → `has_multiple_languages`** : L'EDA montre que 95% des CV ont exactement 1 ou 2 langues. La variable continue n'apporte pas plus de signal qu'une variable binaire, tout en introduisant une skewness artificielle (-1.97). La version binaire est plus robuste et interprétable.
- **Winsorisation (percentile 5-95)** : Appliquée sur `years_experience`, `avg_job_duration`, `nb_certifications`, `nb_technical_skills`, `nb_methods_skills` pour limiter l'influence des valeurs extrêmes avant le calcul des features dérivées.

---

## 4. Comparaison : Modèle IA vs Labels Étudiants

| Source | Taux d'Invitation | Écart de Sévérité |
| :--- | :--- | :--- |
| Labels Étudiants (Heuristique) | 25.5% | Référence |
| Modèle IA (v6) | 30.0% | +4.5 pp |

**Analyse :** Le modèle v6 invite 30% des candidats contre 25.5% pour les labels étudiants, soit un écart de **+4.5 points de pourcentage**. Cet écart reflète le choix délibéré d'un seuil légèrement permissif (0.601) pour préserver un recall minimal sur les profils juniors (Recall=0.222), qui seraient autrement systématiquement exclus par la feature `years_experience`.

---

## 5. Métriques de Performance

### Sur le dataset complet (200 CV)

| Métrique | Valeur |
|---|---|
| **ROC-AUC** | **0.783** |
| Seuil de Tri Optimal | 0.601 |
| Accuracy | 0.765 |
| F1-Score (Invité) | 0.577 |

### Cross-Validation 5-Fold (robustesse)

| Métrique | Moyenne | Écart-type |
|---|---|---|
| **ROC-AUC** | **0.760** | ±0.021 |
| **F1-Score (Invité)** | **0.561** | ±0.059 |

> Pour un outil de **tri** (ranking par probabilité), un ROC-AUC de 0.780 signifie que le modèle classe correctement 78% des paires Invité/Rejeté — performance maintenue par rapport à v5 (0.782) malgré la refonte EDA.

### Matrice de Confusion (dataset complet, 200 CV)

|  | Prédit Rejeté | Prédit Invité |
|---|---|---|
| **Réel Rejeté** | 121 (VN) | 28 (FP) |
| **Réel Invité** | 19 (FN) | 32 (VP) |

![Matrice de Confusion](plots/confusion_matrix.png)

### Courbe ROC

![Courbe ROC](plots/roc_curve.png)

---

## 6. Importance des Variables (SHAP)

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

## 7. Audit d'Équité (Ethics First)

L'audit final, réalisé en fusionnant `features.csv` avec `identities.csv`, donne les résultats suivants :

### Par Genre

| Groupe | n | Recall | Précision |
|---|---|---|---|
| Femmes | 89 | 0.727 | 0.593 |
| Hommes | 111 | 0.552 | 0.485 |

Avantage pour les femmes (+17.5% recall), précision également meilleure. À surveiller sur un dataset plus large.

### Par Âge

| Groupe | n | Recall | Précision |
|---|---|---|---|
| Adulte (30-45) | 132 | 0.714 | 0.536 |
| Jeune (<30) | 68 | 0.222 | 0.500 |

**Amélioration v6 :** Le recall des jeunes (<30) passe de 0.111 (v5) à 0.222 grâce à la feature `junior_potential` (terme d'interaction). 2 juniors sur 9 qui méritent d'être vus sont maintenant identifiés, contre 1 sur 9 précédemment. La précision (0.500) confirme que les juniors sélectionnés sont pertinents.

**Alerte :** Absence totale de profils Senior (>45) dans le dataset — le modèle n'a pas été entraîné sur ce segment.

![Équité](plots/fairness_metrics.png)

### Par Pays (Localité via téléphone)

L'audit par pays révèle des disparités de recall, bien que sur des échantillons réduits (n ~ 20 par pays) :

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

**Analyse :** Les disparités de recall par pays subsistent mais se sont réduites par rapport à v5. Italie et Portugal (Recall=0.25 en v5) atteignent 0.500. Ces écarts reflètent des différences de distribution des variables d'entrée selon les provenances — le modèle ne voit pas le pays.

![Équité par Pays](plots/fairness_country.png)

---

## 8. Conclusion
Ce projet démontre qu'il est possible de concilier performance prédictive et éthique de recrutement. En stabilisant le modèle par la régularisation, en créant des variables de potentiel équitables et en supprimant les variables redondantes, nous fournissons un outil de tri responsable et transparent.

La version v6 apporte un cycle complet d'analyse exploratoire (EDA) intégré au pipeline (`p00_exploration.py`), ayant conduit à deux améliorations concrètes : le log-transform de `avg_job_duration` (skewness 1.61) et la binarisation de `nb_languages` (variable quasi-constante). Ces changements renforcent la robustesse statistique du modèle sans en modifier l'interprétabilité.

Le modèle est conçu comme un **assistant au recruteur** : il priorise les candidatures par score de probabilité et ne prend aucune décision finale autonome.
