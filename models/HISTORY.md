# 📜 Historique des Versions du Modèle - Automatic CV

Ce document retrace les itérations du modèle de tri de CV pour assurer la traçabilité des décisions techniques et éthiques.

## 🟢 v1 : Le Prototype "Généreux" (SMOTE)
- **Architecture :** Logistic Regression + SMOTE (Oversampling).
- **Seuil :** 0.498 (quasi 0.5).
- **Problème :** Précision très faible (45%). Le modèle invitait trop de monde (FP=49).
- **Biais :** Très fort biais "Jeune" (100% d'invitation pour les <30 ans) et Misandrie statistique (Recall plus bas pour les hommes).
- **Leçon :** SMOTE sur un dataset de 200 lignes crée du bruit et floute la frontière de décision.

## 🟡 v2 : Optimisation du Tri (Ranking)
- **Changement :** Suppression de SMOTE, passage à `class_weight='balanced'`.
- **Seuil :** 0.559.
- **Résultat :** Précision remontée à 54%. Meilleure capacité de rejet (88% sur les "Rejetés").
- **Leçon :** La gestion native du déséquilibre est plus stable que la génération de données synthétiques.

## 🔴 v3 : La Tentative de "Fairness" Radicale
- **Changement :** Retrait de `exp_edu_score`, régularisation extrême (`C=0.005`), seuil strict (0.622).
- **Échec :** Le modèle est devenu "muet" (0 invitations). Trop de contraintes sur un petit dataset tuent le signal.
- **Leçon :** On ne peut pas corriger 100% des biais par la contrainte mathématique sans perdre toute utilité métier sur 200 lignes.

## 🔵 v4 : L'Équilibre Fair-Ranking
- **Architecture :** Logistic Regression (`C=0.05`) + Features Robustes.
- **Leçon :** Trop de régularisation et le retrait des scores combinés a fini par discriminer les jeunes par "omission" d'expérience.

## 🔷 v5 : Le "Tri du Potentiel"
- **Changement :** Introduction de `potential_score` (Compétences / (Exp + 1)).
- **Philosophie :** On ne favorise pas les jeunes parce qu'ils sont jeunes, mais parce qu'ils apprennent VITE (densité de skills).
- **Résultats :**
    - ROC-AUC : 0.782 | Seuil : 0.601 | Taux invitation : 26.5%
    - Équité Genre : Recall Femmes=0.591, Hommes=0.552.
    - Recall Jeunes (<30) : 0.111 — structurellement faible (years_experience dominant).
- **Bug découvert :** Le regex de parsing utilisait `-` (tiret court) alors que les CVs utilisent `—` (em dash). Les données v5 provenaient d'un run antérieur au bug. Corrigé en v6.
- **Leçon :** C'est la version de référence pour la capacité discriminante (ROC-AUC).

## 🏆 v7 : Parité Démographique — Correction Biais Âge (Version Actuelle)
- **Date :** 5 juin 2026
- **Problème identifié :** v6 utilisait `best_threshold_recall(target=0.55)` pour le
  seuil junior, qui maximise la précision parmi les seuils atteignant recall ≥ 0.55.
  Résultat : seuil junior (0.474) > seuil adulte (0.460) — les juniors étaient plus
  filtrés que les adultes, à l'inverse de l'intention déclarée.
  De plus, calibrer sur les labels juniors est invalide car `p05_label_audit` confirme
  un biais d'âge significatif dans les labels (p<0.0001 ***).
- **Correction :** Parité démographique (Feldman et al., 2015). Le seuil junior est
  calculé automatiquement pour que le taux d'invitation junior égale le taux adulte.
  Aucune valeur n'est choisie manuellement — le code la produit à chaque run.
- **Résultats :**
    - Seuil adulte : 0.460 (inchangé) | Seuil junior : 0.326 (calculé par parité)
    - Recall junior : 0.556 → **0.833** (+27.7 pp)
    - Écart recall adulte/junior : 32.2 pp → **4.5 pp** (−86%)
    - AUC-ROC : 0.837 (identique — capacité discriminante préservée)
    - Precision junior : 0.303 → 0.136 (trade-off documenté et accepté)
- **Biais géographique :** Aucun biais significatif dans les labels (all ns).
  Recall améliore pour 5/10 pays, aucun ne régresse.
- **Référence :** `docs/audit_comparatif_v6_v7.md`
- **Run MLflow :** `2ee2303cac5740ffbf0729baf1346296`

## 🟢 v6 : L'EDA-Driven + Junior Boost (Archivé)
- **Date :** 14 Avril 2026
- **Changements :**
    - Correction du bug parsing em dash (`—`) dans `p01_parse.py`.
    - Pipeline EDA intégré (`p00_exploration.py`) : outliers IQR, skewness, distributions.
    - Winsorisation (percentile 5-95) sur les comptages auxiliaires (`nb_jobs`, `nb_certifications`, `nb_technical_skills`, `nb_methods_skills`).
    - `nb_languages` → `has_multiple_languages` (binaire, 85% des CV ont exactement 2 langues).
    - Nouvelle feature `junior_potential = is_junior × potential_score` : terme d'interaction permettant à la régression logistique de valoriser différemment le potential_score pour les profils < 3 ans d'expérience.
- **Résultats :**
    - ROC-AUC : 0.783 | CV ROC-AUC : 0.760 ±0.021 (vs ±0.087 en v5 — bien plus stable)
    - Seuil : 0.601 | Taux invitation : 30.0% (+4.5 pp vs labels)
    - Recall Jeunes (<30) : 0.222 (vs 0.111 en v5 — doublement grâce à `junior_potential`)
    - Recall Italie/Portugal : 0.500 (vs 0.250 en v5)
- **Philosophie :** L'EDA révèle, la feature engineering corrige. On ne booste pas les juniors parce qu'ils sont jeunes, mais parce qu'un junior avec un fort `potential_score` mérite un signal additionnel distinct du signal global.
- **Bilan :** Version la plus équitable et la plus stable. ROC-AUC maintenu à 0.783.
