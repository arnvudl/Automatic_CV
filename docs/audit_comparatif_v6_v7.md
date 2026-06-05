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

## 1. Biais de Genre — Avant / Après

### Labels bruts (p05_label_audit — inchangé entre v6 et v7)

| Groupe | N | Invités | Taux | Écart global | p-value | Significatif |
|--------|---|---------|------|-------------|---------|-------------|
| Male   | 267 | 56 | 21.0% | +1.0% | 0.5601 | ns — neutre |
| Female | 233 | 44 | 18.9% | −1.1% | 0.5601 | ns — neutre |

> **Conclusion labels :** Aucun biais de genre statistiquement significatif dans les
> décisions historiques des recruteurs. L'écart de 2.1 pp entre hommes et femmes
> est non significatif (p=0.56).

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
> **Cause de l'amélioration :** Le seuil junior abaissé (0.326) bénéficie
> proportionnellement plus aux femmes si elles sont surreprésentées parmi les juniors
> dans ce dataset — à surveiller sur des données futures.

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

> **Analyse :** Le recall géographique s'améliore pour 5 pays sur 10. Aucun pays
> ne régresse. L'amélioration est plus marquée pour Irlande (+18.2 pp) et
> Pays-Bas (+16.7 pp), probablement corrélée à une proportion de juniors plus
> élevée dans ces groupes.
>
> **Mécanisme indirect :** Le pays n'est pas une feature du modèle. Les disparités
> géographiques observées sont des **biais indirects** (proxy bias) via des
> corrélations avec l'expérience, le niveau d'éducation ou le secteur. Ce biais
> proxy ne peut pas être corrigé par un ajustement de seuil — il nécessiterait
> une intervention au niveau des features ou du dataset.

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
