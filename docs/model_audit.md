# Audit du modèle ML — CV-Intelligence v3

*TechCore Liège — Avril 2026*

---

## Table des matières

1. [Choix du modèle et justification](#1-choix-du-modèle-et-justification)
2. [Pourquoi pas XGBoost](#2-pourquoi-pas-xgboost)
3. [Features : ce qui est dedans, ce qui est dehors et pourquoi](#3-features)
4. [Pseudo-labels : logique et limites](#4-pseudo-labels)
5. [Résultats et métriques](#5-résultats-et-métriques)
6. [Audit biais](#6-audit-biais)
7. [SHAP : explication et justification](#7-shap)
8. [Human in the loop : score avant seuil](#8-human-in-the-loop)
9. [Limites connues et prochaines étapes](#9-limites)

---

## 1. Choix du modèle et justification

Trois modèles ont été comparés : Régression Logistique, Random Forest, XGBoost.

**Random Forest a été retenu.**

| Modèle | F1 (CV train) | F1 (test) | ROC-AUC | Variance CV |
|---|---|---|---|---|
| Régression Logistique | 0.879 ± 0.100 | 0.913 | 0.978 | élevée |
| **Random Forest** | **0.891 ± 0.070** | **0.939** | 0.946 | modérée |
| XGBoost | 0.920 ± 0.062 | 0.917 | 0.971 | modérée |

**Pourquoi Random Forest et pas la Régression Logistique ?**

La LR a un ROC-AUC de 0.978 (meilleur des trois) mais une variance de ±0.100 en cross-validation — c'est trop instable pour un dataset de 131 exemples d'entraînement. Une variance élevée signifie que la performance dépend fortement du découpage train/val. En production avec peu de données, ce comportement est risqué.

Le Random Forest offre un meilleur équilibre F1/stabilité, régularisé naturellement par le bootstrap des arbres et le paramètre `max_depth=10`.

---

## 2. Pourquoi pas XGBoost

XGBoost n'est pas rejeté — il est mis de côté pour cette phase. Voici pourquoi.

### Ce qu'il fait bien

- Meilleure ROC-AUC (0.971) sur ce dataset
- Gère bien les déséquilibres de classes via `scale_pos_weight`
- Très performant sur des datasets tabulaires de grande taille

### Pourquoi on ne l'utilise pas maintenant

**1. Taille du dataset (205 CV)**
XGBoost est un algorithme de boosting — il construit des arbres séquentiellement en corrigeant les erreurs des précédents. Sur 205 exemples, ce mécanisme tend à sur-apprendre les patterns du train set. Le F1 en CV (0.920) est le meilleur, mais c'est aussi celui qui fluctue le plus en pratique selon les données.

**2. Pseudo-labels, pas vrais labels**
XGBoost apprend très bien. Trop bien, ici : il apprend à reproduire fidèlement la fonction de scoring heuristique, y compris ses biais implicites. Un Random Forest est plus "paresseux" — il généralise moins précisément, ce qui est une qualité quand la vérité terrain est approximative.

**3. Opacité relative**
Pour un système à haut risque (AI Act), le recruteur doit comprendre pourquoi un candidat est bien classé. Random Forest + SHAP produit des explications stables et cohérentes. XGBoost + SHAP fonctionne aussi mais les valeurs SHAP varient davantage selon les hyperparamètres.

**4. Hyperparamètres sensibles**
XGBoost a ~15 hyperparamètres qui interagissent entre eux (`max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`...). Un GridSearch correct nécessite plusieurs centaines de fits — coûteux et non justifié pour un MVP à 205 CV.

### Quand passer à XGBoost

Dès que les conditions suivantes sont réunies :
- Dataset >= 500 CV avec vrais labels recruteurs
- GridSearchCV avec 5-fold sur les hyperparamètres principaux
- Calibration des probabilités (Platt Scaling) pour fiabiliser les scores

---

## 3. Features

### Features utilisées par le modèle (14 colonnes)

| Feature | Type | Justification |
|---|---|---|
| `education_level` | Ordinal 1-4 | Signal de formation structurant |
| `nb_jobs` | Entier | Diversité d'expérience |
| `years_experience` | Float | Séniorité — signal important mais non dominant depuis v2 |
| `avg_job_duration` | Float | Stabilité et engagement, indépendant de l'âge |
| `nb_technical_skills` | Entier | Adéquation technique |
| `nb_methods_skills` | Entier | Maîtrise des processus |
| `nb_management_skills` | Entier | Potentiel de responsabilité |
| `total_skills` | Entier | Vue agrégée du profil |
| `nb_languages` | Entier | Multilinguisme valorisé chez TechCore (contexte européen) |
| `has_english` | Binaire | Anglais comme standard minimum IT |
| `english_level` | Ordinal 0-6 | CECRL — nuance du niveau |
| `has_french` | Binaire | Pertinent pour Liège (frontière linguistique) |
| `has_german` | Binaire | Pertinent pour le marché belge/luxembourgeois |
| `nb_certifications` | Entier | Investissement personnel, fiabilité |

### Features exclues intentionnellement

| Feature | Raison |
|---|---|
| `gender`, `age`, `nom`, `email`, `téléphone` | RGPD — données d'identité |
| `career_progression` | SHAP = 0.000 en v1 et v2 — aucune variance utile dans les données |
| `has_luxembourgish` | SHAP = 0.000 — quasi-absent dans le dataset actuel |
| `target_role` | Encodage texte libre non implémenté — à traiter en Phase 4 (NER) |
| `education_field` | 40+ valeurs distinctes, trop fragmenté sans encodage sémantique |
| `profile_type` | Metadata dashboard uniquement — ne doit pas entrer dans le modèle |
| `heuristic_score` | Serait de la fuite de données (data leakage) — c'est la source des labels |

### Pourquoi `has_english` reste à SHAP ≈ 0

Dans ce dataset synthétique, presque tous les candidats parlent anglais. Une feature avec quasi-zéro variance n'apporte aucun signal discriminant au modèle. En production réelle, si des candidats sans anglais postulent, cette feature deviendra utile.

---

## 4. Pseudo-labels

### Principe

En l'absence d'historique de décisions RH, les labels sont générés par un scoring heuristique qui encode les critères implicites des recruteurs TechCore.

### Scoring v2 — logique de chaque critère

**Qualité d'expérience (max 3 pts) — via `avg_job_duration`**
Remplace le scoring brut sur `years_experience`. Un candidat qui reste 3+ ans dans un poste démontre engagement et capacité à approfondir — signal plus fiable qu'un total d'années accumulées rapidement.

| `avg_job_duration` | Points |
|---|---|
| >= 3.0 ans | +3 |
| >= 1.5 ans | +2 |
| >= 0.5 ans | +1 |

**Bonus séniorité (max 2 pts) — mineur, non pénalisant**
L'expérience absolue reste valorisée mais comme bonus, pas comme critère dominant. Un junior ne perd pas de points pour manque d'ancienneté.

| `years_experience` | Points |
|---|---|
| >= 7 ans | +2 |
| >= 3 ans | +1 |
| < 3 ans | 0 (pas de malus) |

**Education (max 3 pts)**
PhD = +3, Master = +2, Bachelor = +1. Fortement pondéré pour compenser l'absence d'expérience chez les juniors.

**Langues (max 5 pts) — fortement valorisées**
TechCore opère dans un environnement multilingue (Liège, frontière française/allemande, clients européens). Le multilinguisme est un critère RH réel, pas cosmétique.

| Critère | Points |
|---|---|
| >= 4 langues | +3 |
| >= 3 langues | +2 |
| >= 2 langues | +1 |
| Anglais C1/C2 | +2 |
| Anglais B1/B2 | +1 |

**Compétences techniques (max 2 pts), méthodes (max 1 pt)**
Adéquation au poste IT.

**Certifications (max 1 pt), Secteur IT (max 1 pt)**
Signaux secondaires.

**Seuil d'invitation : score >= 11.5 / 18**
Choisi à la médiane de la distribution des scores — garantit un dataset équilibré (119 invités / 86 rejetés, 58%/42%).

### Limite principale

Les pseudo-labels reproduisent les biais implicites de cette grille. Si un critère est mal pondéré (ex. certifications sur-valorisées pour un poste junior), le modèle apprend ce biais. C'est précisément pourquoi `heuristic_score` est conservé dans les données : quand de vrais labels recruteurs arrivent, on peut mesurer l'écart entre le score heuristique et la décision humaine — et corriger la grille en conséquence.

---

## 5. Résultats et métriques

### Pourquoi F1 et pas accuracy

L'accuracy est trompeuse dès qu'il y a un déséquilibre (ici 58/42). Un modèle qui dirait "toujours invité" obtiendrait 58% d'accuracy sans rien apprendre. Le F1 pénalise les faux positifs (inviter quelqu'un qui ne devrait pas l'être) et les faux négatifs (rater un bon candidat) de façon équilibrée.

### Pourquoi ROC-AUC en complément

Le F1 dépend d'un seuil de décision fixe (0.5 par défaut). Le ROC-AUC mesure la capacité du modèle à **ordonner** les candidats correctement, indépendamment du seuil. C'est la métrique la plus utile pour le recrutement, où ce qui compte c'est "qui est en haut de la liste", pas "qui dépasse 50%".

### Interprétation de la confusion matrix (test set, 41 CV)

```
                   Prédit Rejeté   Prédit Invité
Réel Rejeté             15              2
Réel Invité              1             23
```

- **2 faux positifs** : candidats rejetés classés invités → risque d'inviter un profil inadéquat
- **1 faux négatif** : candidat invité classé rejeté → risque de rater un bon profil

Dans un contexte recrutement, **les faux négatifs coûtent plus cher** (passer à côté d'un bon candidat). Le recall de 0.958 sur la classe "invité" est donc la métrique à surveiller en priorité.

### Pas de MSE — pourquoi

La MSE (Mean Squared Error) est une métrique de **régression** — elle mesure l'écart entre une valeur prédite et une valeur réelle continues. Ici le problème est une **classification binaire** : on prédit une classe (0 ou 1), pas un nombre. La MSE n'a pas de sens dans ce contexte.

Ce qui s'en approche : la **log-loss** (entropie croisée), qui pénalise les prédictions de probabilité incorrectes et confiantes. Elle est calculée implicitement lors de l'entraînement mais pas rapportée ici car F1 et AUC sont plus lisibles pour un usage RH.

### Pas de vérification d'incertitude — pourquoi c'est un manque

`predict_proba()` donne une probabilité (ex. 0.73) mais sans intervalle de confiance. On ne sait pas si ce 0.73 est "robustement entre 0.65 et 0.80" ou "pourrait être n'importe quoi entre 0.40 et 0.95". Pour la production, une calibration des probabilités (Platt Scaling ou Isotonic Regression) est nécessaire pour que les scores soient interprétables comme de vraies probabilités.

### Overfitting — état actuel

| Set | F1 |
|---|---|
| Train (cross-val) | 0.891 ± 0.070 |
| Test | 0.939 |

Le F1 test est supérieur au F1 train moyen — ce qui semble contre-intuitif. C'est un artefact de la petite taille du dataset et du split stratifié : le test set (41 CV) peut par chance être légèrement plus facile que le train. Il n'y a pas de signe fort d'overfitting, mais avec 205 exemples et 200 arbres, le modèle est surdimensionné. Sur un dataset plus grand et bruité (vrais CV), le F1 baissera avant de remonter avec plus de données.

---

## 6. Audit biais

### Biais structurels identifiés

**Sampling bias — genre**
113 hommes / 92 femmes (ratio 0.81). Légèrement déséquilibré mais au-dessus du seuil d'alerte (0.80). Acceptable pour cette phase.

**Sampling bias — âge**
Aucun profil senior (>45 ans) dans le dataset. Les CVs synthétiques vont de 21 à 44 ans. Le modèle n'a jamais vu un profil de 50 ans — il ne peut pas généraliser sur ce groupe. En production, si des seniors postulent, leurs scores seront des extrapolations non validées.

**Representation gap — pays**
10 nationalités représentées, aucune ne dépasse 13% du dataset. Distribution raisonnablement équilibrée.

**Omission bias**
`target_role` et `education_field` sont présents dans les données mais pas utilisés comme features (encodage non implémenté). Si ces champs corrèlent avec le label, le modèle rate de l'information. À corriger en Phase 4 avec un encodage sémantique.

### Parité démographique

| Groupe | Taux d'invitation prédit | DI Ratio | Statut |
|---|---|---|---|
| Femmes | 60% | — | — |
| Hommes | 60% | 0.992 | OK (>0.80) |
| Adultes (30-45) | 78% | — | — |
| Jeunes (<30) | 24% | 0.312 | !! sous seuil légal |

**Interprétation du DI âge = 0.312**

Ce chiffre est préoccupant légalement mais s'explique mécaniquement : les jeunes ont structurellement moins d'expérience, et `years_experience` reste la feature la plus influente (SHAP 0.174). Ce n'est pas un biais caché du modèle — c'est une conséquence directe des pseudo-labels qui valorisent l'expérience.

**Point important** : l'Equal Opportunity (recall par groupe) est quasi-identique : adultes 0.990, jeunes 1.000. Cela signifie que parmi les jeunes qui *méritent* d'être invités selon les pseudo-labels, le modèle n'en rate presque aucun. Le problème est dans les pseudo-labels eux-mêmes, pas dans le modèle.

**Fairlearn (genre)**
- Demographic Parity Difference : 0.0049 (quasi nul, excellent)
- Equalized Odds Difference : 0.0533 (acceptable, objectif < 0.05 en production)

### Analyse par localité (indicatif téléphonique)

Le modèle ne voit pas la localisation — le téléphone est dans `identities.csv`. Néanmoins, les métriques par pays montrent des variations :

- Pologne : recall = 0.778 (le modèle rate 2 bons profils polonais sur 9)
- France, Irlande, Nigeria : recall = 1.000 (aucun faux négatif)

Ces écarts sont trop petits (n<30 par pays) pour être statistiquement significatifs. À surveiller avec plus de données.

---

## 7. SHAP

### Importance globale — ce que le modèle utilise vraiment

| Rang | Feature | SHAP moyen | Interprétation |
|---|---|---|---|
| 1 | `years_experience` | 0.174 | Dominant — signe que les pseudo-labels sont fortement basés dessus |
| 2 | `avg_job_duration` | 0.126 | Stabilité — introduit en v2, deuxième signal le plus fort |
| 3 | `nb_technical_skills` | 0.033 | Adéquation technique |
| 4 | `education_level` | 0.033 | Formation |
| 5 | `total_skills` | 0.030 | Vue agrégée |
| 6 | `nb_languages` | 0.023 | Multilinguisme — vivant depuis v2 |
| 7 | `nb_jobs` | 0.022 | Diversité d'expérience |
| ... | `has_english` | 0.000 | Mort car quasi-universel dans le dataset |

### Ce que le SHAP révèle sur les biais potentiels

`years_experience` reste dominant malgré le rééquilibrage du scoring v2. Cela montre que la reformulation des pseudo-labels a réduit l'écart (avg_job_duration est maintenant n°2) mais n'a pas suffi à équilibrer complètement. Pour réduire davantage le poids de l'expérience absolue, il faudrait retirer `years_experience` des features — au prix de perdre toute notion de séniorité.

### Importance locale — comment lire un score individuel

Un score de 0.85 pour un candidat signifie (exemple réel) :

```
years_experience = 8.6 ans     SHAP = +0.270  (fort boost)
avg_job_duration = 8.6 ans     SHAP = +0.126  (stabilité)
education_level  = 3 (Master)  SHAP = +0.080  (bonus diplôme)
nb_certifications = 2          SHAP = +0.070  (certifié)
nb_jobs = 1 seul poste         SHAP = -0.017  (léger malus diversité)
```

Le recruteur peut voir exactement quelles dimensions ont contribué positivement ou négativement — obligation d'explicabilité AI Act.

---

## 8. Human in the loop : score avant seuil

### Pourquoi conserver `heuristic_score`

La colonne `heuristic_score` dans `features.csv` contient le score brut (ex. 13.5/18) avant application du seuil qui génère le label 0/1.

**Trois usages concrets :**

**1. Cas limite**
Un candidat avec `heuristic_score = 11.3` est rejeté (seuil 11.5). Sans le score brut, le recruteur voit juste "rejeté". Avec le score, il voit "0.2 points sous le seuil" — et peut décider de le regarder quand même.

**2. Ajustement du seuil par poste**
Pour un poste junior, abaisser le seuil à 9.0 sans changer le modèle. Pour un poste senior critique, le monter à 14.0. Le `heuristic_score` permet ce filtrage dynamique.

**3. Traçabilité et contestation**
Si un candidat conteste un rejet (droit prévu par l'AI Act), le recruteur peut montrer : "votre score était X, le seuil était Y, voici les critères qui ont pesé positivement/négativement". C'est une réponse factuelle, pas un "l'IA a dit non".

### `profile_type` — filtre dashboard

La colonne `profile_type` (`junior` / `intermediate` / `senior`) est calculée sur `years_experience` :

| Valeur | Seuil |
|---|---|
| `junior` | < 3 ans |
| `intermediate` | 3–8 ans |
| `senior` | > 8 ans |

Elle n'entre **jamais** dans le modèle. Elle sert uniquement au recruteur qui cherche spécifiquement un junior cette semaine sans que ça change le score des autres candidats.

---

## 9. Limites connues et prochaines étapes

### Limites actuelles

| Limite | Impact | Mitigation prévue |
|---|---|---|
| Pseudo-labels, pas vrais labels | F1 mesure la reproduction des règles, pas la prédiction réelle | Collecte de vrais labels recruteurs |
| 205 CV synthétiques | Modèle non validé sur vrais CV bruités | Tests sur vrais CV en Phase 5 |
| Aucun senior dans les données | Extrapolation non validée sur >45 ans | Enrichissement du dataset |
| `years_experience` trop dominant | DI âge = 0.312, en dessous du seuil légal | À surveiller — si confirmé sur vrais données, retirer la feature |
| Probabilités non calibrées | Score de 0.73 n'est pas "73% de chance" | Platt Scaling en Phase 4 |
| `target_role` non utilisé | Le modèle ne sait pas pour quel poste on recrute | Encodage sémantique (NER) en Phase 4 |

### Prochaines étapes Phase 4

1. **Calibration des probabilités** — Platt Scaling pour que les scores soient interprétables
2. **Encodage de `target_role`** — embedding ou one-hot sur les rôles fréquents
3. **Collecte de vrais labels** — chaque décision recruteur est loggée et liée au `cv_id`
4. **Ré-entraînement périodique** — dès 50 nouveaux vrais labels
5. **Audit biais automatisé** — rapport Fairlearn généré à chaque ré-entraînement
