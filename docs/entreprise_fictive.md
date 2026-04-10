# TechCore Liège — Contexte entreprise fictive

## L'entreprise

**TechCore** est une entreprise IT belge de **1 000 employés** basée à **Liège**.  
Elle développe des logiciels B2B pour le secteur industriel (ERP, MES, supervision d'usines) et emploie principalement des profils techniques : développeurs, data engineers, chefs de projet IT, DevOps, architectes.

Le département RH reçoit en moyenne **150 candidatures par mois** pour des postes tech. L'équipe recrutement est composée de 4 personnes. Chaque CV est lu manuellement, ce qui prend entre 5 et 10 minutes. Résultat : **600 à 1 500 heures de travail RH par an** consacrées au premier tri, avant même un entretien.

L'objectif du projet CV-Intelligence est d'automatiser ce premier tri tout en restant conforme RGPD et AI Act.

---

## Le problème des labels

Pour entraîner un modèle supervisé, il faudrait des milliers de CVs avec la mention **"invité en entretien"** ou **"rejeté"** associée.

TechCore n'a pas historisé ces décisions de manière structurée. Les anciens CV sont dans des boîtes mail ou des dossiers partagés, sans tag systématique. Il n'existe pas de base de données des décisions passées.

**On ne peut donc pas constituer un jeu de données labellisé à partir de l'historique.**  
C'est la situation la plus courante dans les PME et ETI.

---

## La solution : bootstrapping par pseudo-labels

Le bootstrapping est une technique qui permet de **démarrer sans label** en générant une vérité terrain synthétique à partir de règles métier.

### Principe

```
Règles métier (expertes)
        ↓
   Score heuristique
        ↓
   Seuil de coupure
        ↓
  Pseudo-label (0 ou 1)
        ↓
Entraînement du modèle ML
        ↓
  Prédictions sur nouveaux CVs
        ↓
Validation humaine (recruteur)
        ↓
  Remplacement progressif
  des pseudo-labels par vrais labels
        ↓
   Modèle qui s'améliore
```

### Pourquoi ça fonctionne

Les règles métier encodent la **connaissance implicite** des recruteurs : "on cherche un Bachelor minimum avec 3 ans d'expérience et l'anglais courant". Cette connaissance existe — elle n'est juste jamais formalisée.

Le modèle ML n'apprend pas à la place du recruteur. Il **apprend à reproduire les règles métier**, puis les généralise sur des profils atypiques que les règles ne savent pas bien traiter (ex : quelqu'un sans diplôme mais avec 10 ans d'expérience et 15 certifications).

Au fil du temps, les vrais retours des recruteurs (invité / rejeté) remplacent les pseudo-labels. Le modèle s'affine et dépasse progressivement les règles de départ.

---

## Règles métier retenues pour TechCore

TechCore recrute principalement sur des postes IT (dev, data, DevOps, architecture). Les critères retenus par l'équipe RH sont les suivants :

| Critère | Condition | Points |
|---|---|---|
| Expérience | >= 5 ans | +3 |
| Expérience | >= 3 ans | +2 |
| Expérience | >= 1 an | +1 |
| Niveau d'études | Master ou plus (level >= 3) | +2 |
| Niveau d'études | Bachelor (level = 2) | +1 |
| Anglais | Présent | +1 |
| Anglais courant | Level >= C1 (level >= 5) | +1 |
| Compétences techniques | >= 5 skills | +2 |
| Compétences techniques | >= 3 skills | +1 |
| Compétences méthodes | >= 3 | +1 |
| Compétences management | >= 2 | +1 |
| Certifications | >= 2 | +1 |
| Certifications | >= 1 | +0.5 |
| Progression de carrière | Détectée | +1 |
| Secteur | IT | +1 |
| Langues | >= 2 langues | +1 |

**Seuil d'invitation** : score >= 8 → label `1` (invité), sinon `0` (non retenu)

Score maximum théorique : ~16 points.

---

## Limites et précautions

- Les pseudo-labels ne sont **pas la vérité** — ils sont une approximation des règles RH actuelles.
- Un modèle entraîné sur des pseudo-labels reproduit les biais implicites de ces règles (ex : sur-pondération du diplôme).
- C'est pourquoi l'**audit biais (Fairlearn)** est prévu en Phase 4 : on vérifiera si le modèle désavantage systématiquement certains groupes.
- Toute décision du modèle reste soumise à révision humaine (obligation AI Act).

---

*Entreprise fictive créée à des fins de développement et de test du système CV-Intelligence.*  
*Les règles métier ci-dessus sont des hypothèses de travail, à valider avec de vrais recruteurs avant mise en production.*
