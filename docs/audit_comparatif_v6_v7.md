# Audit Comparatif — Modèle v6 → v7
**Date :** 5 juin 2026  
**Auteur :** Luminary ATS — Pipeline ML  
**Runs MLflow :** `8a9070d13dae` (v6) → `2ee2303cac57` (v7)

---

## Contexte et motivation

Le modèle v6 utilisait un double seuil âge (adulte/junior) avec comme objectif
`recall_junior ≥ 0.55`. L'audit a révélé un **défaut de conception** :

- La fonction `best_threshold_recall` retourne le seuil **maximisant la précision**
  parmi ceux atteignant recall ≥ 0.55 — soit le seuil le plus **restrictif** acceptable.
- Résultat : seuil junior (0.474) > seuil adulte (0.460), pénalisant les juniors
  en pratique (un junior scoring 0.462 échoue, un adulte scoring 0.462 passe).
- Le biais d'âge dans les **labels** (`p05_label_audit` : p<0.0001 ***) rend toute
  calibration sur les labels juniors non fiable — les "vrais positifs" sont eux-mêmes
  un sous-ensemble biaisé.

**Correction v7 :** Parité démographique (Feldman et al., 2015).  
Le seuil junior est calculé pour que le **taux d'invitation junior = taux d'invitation
adulte observé**, sans se fonder sur les labels biaisés.

---

## 1. Biais de Genre — Évolution et Correction

### Chronologie du biais de genre à travers les versions

Le biais de genre n'existe pas dans les labels bruts, mais a été introduit puis
corrigé par les choix de features successifs :

| Version | Recall Femme | Recall Homme | Écart | Feature responsable                                                                    |
|---------|-------------|-------------|-------|----------------------------------------------------------------------------------------|
| Labels bruts | 18.9% (invite) | 21.0% (invite) | +2.1pp H>F | Décisions humaines — légèrement misogyne                                               |
| **v5** | 59.1% | 55.2% | +3.9pp F>H | `years_experience` dominant → pénalise les carrières fragmentées → légèrement misandre |
| **v6** | 81.8% | 82.1% | +0.3pp H>F ≈ égal | `exp_per_year_of_age` remplace `years_experience` → quasi-parité                       |
| **v7** | 90.9% | 83.9% | +7.0pp F>H | Correction parité démo âge → légèrement misandre                                       |

### Ce qui a corrigé le biais de genre : `exp_per_year_of_age`

**v5 utilisait `years_experience` comme feature principale (SHAP = 0.52 — dominant).**
Cette feature pénalise toute personne ayant des interruptions de carrière, favorisant
mécaniquement les profils avec une ancienneté brute élevée.

**v6 a remplacé `years_experience` par `exp_per_year_of_age` :**
```
exp_per_year_of_age = years_experience / max(age - 22, 1)
```
Cette normalisation mesure l'expérience **relative à la durée de carrière théorique** :
- Un candidat de 27 ans avec 5 ans d'exp → ratio = 5/5 = 1.0
- Un candidat de 37 ans avec 5 ans d'exp → ratio = 5/15 = 0.33 (pénalisé)

**Effet mesuré sur le dataset (vérifié) :**

| Feature | Moy. Femmes | Moy. Hommes | Écart |
|---------|------------|------------|-------|
| `exp_per_year_of_age` | 0.652 | 0.601 | **+8.6% F>H** |
| `has_multiple_languages` | 0.906 | 0.850 | +6.5% F>H |
| `junior_potential` | 1.004 | 1.090 | −7.9% F<H |

Les femmes du dataset ont un `exp_per_year_of_age` 8.6% plus élevé que les hommes.
Ce n'est pas un choix délibéré : c'est la réalité du dataset. En remplaçant
`years_experience` par `exp_per_year_of_age`, le modèle est passé de légèrement
misogyne à quasi-paritaire (v6 : écart 0.3 pp).

**L'écart résiduel v7 (+7.0 pp F>H)** vient de la correction de parité démographique
pour l'âge : les femmes adultes ayant un taux d'invitation historique légèrement plus
bas (23.8% vs 26.0% hommes), le seuil junior calculé par parité leur est
proportionnellement plus favorable.

### Labels bruts (p05_label_audit)

| Groupe | N | Invités | Taux | Écart global | p-value | Significatif |
|--------|---|---------|------|-------------|---------|-------------|
| Male   | 267 | 56 | 21.0% | +1.0% | 0.5601 | ns — neutre |
| Female | 233 | 44 | 18.9% | −1.1% | 0.5601 | ns — neutre |

> **Conclusion labels :** Aucun biais de genre statistiquement significatif dans les
> décisions historiques (p=0.56).

### Performances du modèle

| Métrique | Modèle v6 (seuil junior=0.474) | Modèle v7 (seuil junior=0.326) | Évolution |
|----------|-------------------------------|-------------------------------|-----------|
| Recall Female | 0.818 | **0.909** | +9.1 pp ✅ |
| Recall Male   | 0.821 | **0.839** | +1.8 pp |
| Écart M−F | +0.3 pp | −7.0 pp | Inversion |
| Precision Female | 0.340 | 0.274 | −6.6 pp |
| Precision Male   | 0.371 | 0.292 | −7.9 pp |

> **Analyse :** Le recall global progresse pour les deux genres. L'écart s'inverse
> légèrement en faveur des femmes (−7.0 pp), restant dans la zone acceptable (≤10 pp).
> La baisse de précision est le coût direct de la correction de parité démographique.
>
> **Analyse de l'origine de l'écart (vérifié sur données) :**
> La distribution juniors/adultes est quasi-identique entre genres :
> Femmes 26.2% juniors, Hommes 26.6% juniors — la surreprésentation féminine
> parmi les juniors n'explique PAS l'écart.
>
> Dans les labels bruts, les femmes juniors sont encore plus défavorisées que les hommes
> juniors (taux invitation 4.9% vs 7.0%). La correction de parité démographique
> bénéficie légèrement plus aux femmes adultes (23.8% invite rate) qu'aux hommes
> adultes (26.0%), ce qui crée mécaniquement un léger avantage recall femmes.
>
> **L'écart de 7 pp n'est pas dû aux features ni à la composition age/genre
> du dataset — il provient du taux d'invitation historique légèrement plus bas
> pour les femmes adultes (23.8% vs 26.0%), qui se traduit par un seuil junior
> proportionnellement plus favorable aux femmes.**

---

## 1b. Parité Démographique — Principe et Implémentation

### Pourquoi pas Equal Opportunity ?

L'approche intuitive pour corriger le biais d'âge serait **Equal Opportunity** :
fixer le seuil junior pour que recall junior = recall adulte. Mais cette approche
est invalide ici car les labels d'entraînement sont eux-mêmes biaisés (p<0.0001).

Calibrer un seuil sur des labels biaisés revient à apprendre à reproduire le biais
à une fréquence "acceptable". Les "vrais positifs" juniors dans le dataset sont déjà
un sous-ensemble filtré par des recruteurs biaissés — les juniors exceptionnels qui
ont surmonté le biais humain. Atteindre un recall égal sur ce sous-ensemble ne
corrige pas le problème de fond.

### La parité démographique (Feldman et al., 2015)

**Principe :** fixer le seuil junior de sorte que le taux d'invitation junior
**calculé par le modèle** égale le taux d'invitation adulte **observé dans les données**.

```
taux_adulte_observé = P(score_adulte ≥ seuil_adulte) = 62.3% (sur train)
seuil_junior = score minimal tel que P(score_junior ≥ seuil_junior) = 62.3%
             = top 62.3% des scores juniors triés par ordre décroissant
             = 0.326
```

**Ce que cette approche garantit :**
- Le seuil junior est **calculé automatiquement** à chaque ré-entraînement
- Il ne repose **pas** sur les labels juniors (qui sont biaisés)
- Il est entièrement déterminé par la distribution des scores et le comportement
  adulte — deux grandeurs non biaisées par les décisions historiques anti-junior
- Il est reproductible, auditable, et ancré dans la littérature de fairness

**Ce que cette approche ne garantit pas :**
- Elle n'efface pas le biais historique dans les labels
- Elle réduit l'écart de recall (32.2 pp → 4.5 pp) mais ne l'annule pas
- Elle introduit un trade-off précision (documenté et accepté)

---

## 1c. Analyse SHAP — Importance des Features

SHAP (SHapley Additive exPlanations) mesure la contribution de chaque feature
au score individuel d'un candidat. Ci-dessous les valeurs d'importance globale
(moyenne des |valeurs SHAP| sur le dataset de test) du modèle v7.

### Importance globale des features

| Rang | Feature | Importance SHAP | Rôle dans les biais |
|------|---------|----------------|-------------------|
| 1 | `education_adj` | **0.2392** | Biais académique — Master favorisé. Compressé (0.30/0.70) vs `education_level` brut (SHAP=0.52 en v1) pour réduire sur-pondération diplôme |
| 2 | `career_depth` | **0.1550** | **Proxy géographique** — France/Portugal ont career_depth hors-norme. Non corrigible par seuil |
| 3 | `potential_score` | **0.1223** | Anti-biais âge — valorise compétences/expérience, favorable aux juniors avec beaucoup de skills |
| 4 | `junior_potential` | **0.1007** | Terme d'interaction IS_junior × potential_score — signal additionnel pour juniors à fort potentiel |
| 5 | `avg_job_duration` | **0.0908** | Corrélé à career_depth — même proxy géographique indirect |
| 6 | `has_multiple_languages` | **0.0788** | Légèrement favorable aux femmes (+6.5% F>H dans le dataset) |
| 7 | `field_match` | **0.0479** | Adéquation formation/secteur — neutre sur biais démographiques |
| 8 | `exp_per_year_of_age` | **0.0385** | **Fix genre** — remplace `years_experience`, neutralise le biais carrières fragmentées. Faible importance globale mais fort impact sur l'équité |
| 9 | `is_it` | **0.0347** | Secteur IT — neutre, avantage structurel IT vs Finance déjà dans les labels |

### Lecture des biais à travers SHAP

**Biais d'âge :** `potential_score` (rang 3) et `junior_potential` (rang 4) ont été
introduits spécifiquement pour valoriser les juniors à fort potentiel sans utiliser
l'âge comme feature directe. Ensemble ils représentent **22.3% de l'importance totale**.

**Biais de genre :** `exp_per_year_of_age` (rang 8, 3.85%) a un impact faible en
importance globale mais décisif sur l'équité : c'est lui qui a cassé la dominance
de `years_experience` (SHAP=0.52 en v5) et ramené l'écart genre de 3.9 pp à 0.3 pp.

**Biais géographique :** `career_depth` (rang 2, 15.5%) et `avg_job_duration` (rang 5,
9.1%) sont les vecteurs proxy des disparités géographiques. France et Portugal ont
des career_depth atypiques (respectivement −4.58 et +4.16 vs moyenne) qui se
traduisent en scores défavorables, sans que le pays soit une feature directe.

> ⚠️ `education_adj` reste la feature la plus influente (23.9%). Le biais académique
> (Master 30.1% vs Bachelor 12.7% d'invitation dans les labels) est atténué par
> la compression de l'échelle mais pas éliminé. C'est le biais le plus difficile
> à corriger sans relabeling humain des cas litigieux.

---

## 2. Biais d'Âge — Avant / Après

### Labels bruts (p05_label_audit)

| Groupe | N | Invités | Taux | Écart global | p-value | Significatif |
|--------|---|---------|------|-------------|---------|-------------|
| Adulte (30–45) | 326 | 82 | 25.2% | +5.2% | 0.0001 | *** — favorisé |
| Junior (<30)   | 174 | 18 | 10.3% | −9.7% | 0.0001 | *** — défavorisé |
| Senior (>45)   |   0 |  0 |   0%  | —     | —      | ⚠️ absent du dataset |

> **Conclusion labels :** Biais d'âge **confirmé et hautement significatif** (p<0.0001).
> Les adultes (30–45 ans) sont invités à 25.2% vs 10.3% pour les juniors, soit
> un écart de **14.9 points de pourcentage**. Ce biais est encodé dans les labels
> et sera reproduit par tout modèle entraîné dessus sans correction.
>
> ⚠️ **Absence totale de profils Senior (>45 ans) dans le dataset.** Le modèle
> ne peut pas être évalué sur ce groupe. Toute décision automatisée sur des seniors
> doit être revue manuellement.

### Seuils appliqués

| Version | Seuil adulte | Seuil junior | Junior < Adulte ? |
|---------|-------------|-------------|-------------------|
| v6 | 0.460 | 0.474 | ❌ Junior plus strict |
| v7 | 0.460 | **0.326** | ✅ Junior plus permissif |

### Performances du modèle

| Métrique | Modèle v6 | Modèle v7 | Évolution |
|----------|-----------|-----------|-----------|
| Recall Adulte | 0.878 | 0.878 | = inchangé |
| Recall Junior | 0.556 | **0.833** | +27.7 pp ✅ |
| Écart Adulte−Junior | **32.2 pp** | **4.5 pp** | −27.7 pp ✅ |
| Precision Adulte | 0.365 | 0.365 | = inchangé |
| Precision Junior | 0.303 | 0.136 | −16.7 pp ⚠️ |

> **Analyse :** La correction par parité démographique réduit l'écart de recall
> de 32.2 pp à 4.5 pp — une réduction de **86%** de la disparité.
>
> **Trade-off documenté :** La précision junior chute de 30.3% à 13.6%. Cela signifie
> qu'un plus grand nombre de juniors invités ne seront pas retenus en entretien.
> Ce coût est **délibérément accepté** car :
> 1. La précision est mesurée sur des labels historiquement biaisés contre les juniors.
>    Les "faux positifs" juniors sont potentiellement de bons candidats injustement
>    rejetés par les recruteurs humains passés.
> 2. Le rôle du système est de **pré-filtrer**, non de décider. L'entretien reste
>    la vérification humaine finale.
> 3. L'AI Act (Art. 9, 10) impose de documenter et justifier les choix de conception
>    impactant des groupes protégés — ce choix l'est.

---

## 3. Biais Géographique

### Labels bruts (p05_label_audit) — aucun pays utilisé comme feature ML

| Pays | N | Taux invitation | Écart | p-value | Significatif |
|------|---|----------------|-------|---------|-------------|
| Pays-Bas | 45 | 27.7% | +7.7% | 0.168 | ns |
| Irlande | 36 | 25.0% | +5.0% | 0.436 | ns |
| USA/Canada | 49 | 24.6% | +4.6% | 0.166 | ns |
| Pologne | 57 | 24.5% | +4.5% | 0.408 | ns |
| Inde | 42 | 23.8% | +3.8% | 0.519 | ns |
| Italie | 46 | 21.7% | +1.7% | 0.757 | ns |
| Nigeria | 61 | 11.9% | −8.1% | 0.171 | ns |
| France | 50 | 11.8% | −8.2% | 0.121 | ns |
| Portugal | 39 | 10.3% | −9.7% | 0.144 | ns |
| Allemagne | 34 | 8.8% | −11.2% | 0.119 | ns |

> **Conclusion labels :** Aucun biais géographique **statistiquement significatif**
> dans les labels. Les écarts observés (jusqu'à 11 pp pour l'Allemagne) pourraient
> être des artefacts du faible effectif par pays (34–61 candidats).
>
> ⚠️ **Pattern à surveiller :** Nigeria, France, Portugal et Allemagne apparaissent
> systématiquement en bas du classement. Bien que non significatif sur ce dataset,
> ce pattern mérite une attention accrue si le volume de données augmente.

### Performances du modèle par pays — v6 vs v7

| Pays | Recall v6 | Recall v7 | Évolution |
|------|-----------|-----------|-----------|
| Allemagne | 0.917 | 1.000 | +8.3 pp ✅ |
| Pologne | 1.000 | 1.000 | = |
| Portugal | 0.900 | 0.900 | = |
| Inde | 0.833 | 0.833 | = |
| Italie | 0.857 | 0.857 | = |
| USA/Canada | 0.800 | 0.800 | = |
| Pays-Bas | 0.833 | 1.000 | +16.7 pp ✅ |
| Irlande | 0.727 | 0.909 | +18.2 pp ✅ |
| Nigeria | 0.733 | 0.733 | = |
| France | 0.700 | 0.800 | +10.0 pp ✅ |

> ⚠️ **Distinction essentielle : recall ≠ taux d'invitation**
>
> Ce que v7 améliore : le **recall par pays** — ne pas rater les candidats
> positifs qui existent dans les données (ex. Allemagne recall 91.7% → 100%).
>
> Ce que v7 ne corrige **pas** : l'**écart de taux d'invitation entre pays**.
> Les Pays-Bas avaient 27.7% d'invitation dans les labels, l'Allemagne 8.8%.
> Le modèle apprend à bien identifier ces 8.8% d'Allemands positifs — mais
> l'écart de 18.9 pp avec les Pays-Bas reste entier dans les données sources.
>
> **Pour corriger le taux**, il faudrait appliquer la parité démographique par
> pays : forcer le modèle à inviter les Allemands au même taux que les Néerlandais.
> Cette correction n'a **pas** été appliquée pour deux raisons :
>
> 1. **Absence de biais prouvé dans les labels** (all ns, p>0.10) — les écarts
>    de taux pourraient être du bruit statistique lié aux faibles effectifs
>    (34–61 par pays), pas un biais systématique
> 2. **Effectifs insuffisants** — avec 3 vrais positifs allemands sur 34,
>    appliquer une parité démographique forcerait une égalisation statistiquement
>    non fiable et potentiellement arbitraire
>
> **Conclusion géographique :** Le biais géographique reste une **limitation ouverte**.
> Il nécessite d'abord une preuve statistique de biais dans les labels (p<0.05),
> puis des effectifs suffisants (>100/pays) pour une correction valide.
>
> **Pourquoi aucune correction n'a été appliquée :**
> - Aucun biais géographique statistiquement significatif dans les labels (all ns)
> - Effectifs trop faibles (34–61 par pays) pour une parité démographique par pays
>   statistiquement valide
> - Le pays n'est pas une feature du modèle — le biais est indirect (proxy bias)
>   via des corrélations avec l'expérience, l'éducation et le secteur
>
> **Ce qu'il faudrait pour le corriger :**
> 1. Augmenter le dataset (> 100 candidats par pays minimum)
> 2. Analyser les corrélations pays ↔ features pour identifier le vecteur proxy
> 3. Appliquer une parité démographique par pays si les effectifs le permettent
>
> **Statut : surveillance active — correction reportée faute de données suffisantes.**

---

## 4. Synthèse des Seuils

| Version | Seuil adulte | Seuil junior | Méthode | Justification |
|---------|-------------|-------------|---------|--------------|
| v6 | 0.460 | 0.474 | recall ≥ 0.55 | ❌ Junior plus strict — contre-intuitif |
| **v7** | **0.460** | **0.326** | **Parité démographique** | ✅ Ancré dans les données, non arbitraire |

> Le seuil 0.326 est **calculé automatiquement** à chaque ré-entraînement comme le
> score minimal permettant d'inviter les top-N% juniors, N étant le taux d'invitation
> adulte observé (62.3% sur le train set). Il n'est **jamais choisi manuellement**.

---

## 5. Métriques Globales

| Métrique | v6 | v7 | Évolution |
|----------|----|----|-----------|
| AUC-ROC | 0.837 | 0.837 | = inchangé |
| Accuracy | 0.72 | 0.59 | −13 pp ⚠️ |
| Recall global (Invités) | 0.90 | 0.90 | = |
| Precision globale (Invités) | 0.41 | 0.32 | −9 pp |
| F1 global | 0.56 | 0.47 | −9 pp |

> L'AUC-ROC reste identique (0.837) — le modèle conserve sa capacité discriminante.
> La baisse d'accuracy et de F1 est le coût de la correction de fairness.
> Ce trade-off est documenté, justifié et accepté.

---

## 6. Points de Vigilance et Limites

1. **Absence de seniors (>45 ans) :** Le modèle n'a jamais vu de profils seniors.
   Toute décision automatisée sur ce groupe est non validée.

2. **Label bias irréductible :** Le biais d'âge dans les labels (p<0.0001) ne peut
   pas être éliminé par un ajustement de seuil. La correction démographique en
   atténue les effets mais ne le supprime pas.

3. **Effectifs géographiques faibles :** 34 à 61 candidats par pays — trop peu pour
   des conclusions statistiques solides. À réévaluer avec plus de données.

4. **Seuil dynamique :** Le seuil junior (0.326) variera à chaque ré-entraînement
   selon la distribution des scores. C'est une propriété souhaitée (auto-calibration)
   mais elle doit être monitorée (MLflow log systématique).

5. **Supervision humaine obligatoire :** Ce système est un outil d'aide à la décision.
   Toute décision finale reste sous responsabilité humaine (AI Act Art. 14).

---

*Document généré automatiquement depuis les artefacts MLflow.*  
*Run v6 : `8a9070d13dae4d169b19582944a45d7c` | Run v7 : `2ee2303cac5740ffbf0729baf1346296`*
