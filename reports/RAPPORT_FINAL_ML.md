# Rapport de Projet : Système de Tri de CV Automatisé (v5)

**Candidats :** Tom Perez Le Tiec | Arnaud Leroy  
**Date :** 13 Avril 2026  
**Version du Modèle :** v5 (Potential & Fair Optimized)

---

## 1. Synthèse du Projet
L'objectif de ce projet est de concevoir un système de tri (ranking) de CV capable d'identifier les profils les plus pertinents pour un processus de recrutement. Contrairement à un système de décision automatique, ce modèle sert d'assistant au recruteur pour prioriser l'examen des candidatures.

Une attention particulière a été portée à l'éthique algorithmique. Nous avons fait le choix délibéré de privilégier la neutralité et la réduction des biais, même si cela a nécessité un compromis sur la performance brute du modèle.

---

## 2. Architecture Technique

### Modèle Utilisé
Le modèle retenu est une Régression Logistique avec une forte régularisation (C=0.05). Ce choix se justifie par la stabilité du modèle sur un petit dataset (200 CV) et sa capacité à fournir des probabilités claires pour le classement. La gestion du déséquilibre des classes (75% Rejetés / 25% Invités) est assurée nativement par le paramètre class_weight='balanced'.

### Variables Utilisées (Robust Features)
Nous avons sélectionné 8 variables clés, choisies pour leur stabilité et leur pertinence métier :
- years_experience : Nombre total d'années d'expérience.
- avg_job_duration : Durée moyenne par poste (mesure de stabilité).
- education_level : Niveau d'études (encodé de 1 à 5).
- potential_score : Score de potentiel (voir Feature Engineering).
- nb_languages : Nombre de langues maîtrisées.
- career_depth : Profondeur de carrière (Expérience * Durée moyenne).
- is_it : Indicateur du secteur Informatique.
- is_finance : Indicateur du secteur Finance.

---

## 3. Feature Engineering
L'étape d'ingénierie des variables a été le pivot de notre approche éthique.

- Suppression du score combiné (exp_edu_score) : Initialement utilisé, ce score multipliait l'expérience par le diplôme, ce qui créait un biais massif en faveur des profils installés et excluait systématiquement les jeunes talents.
- Introduction du Potential Score : Nous avons créé la variable potential_score = (Compétences techniques + Méthodes + Certifications) / (Années d'expérience + 1). Cette variable permet de valoriser la "densité" de compétences et la vitesse d'apprentissage, offrant ainsi une chance aux profils juniors à haut potentiel de remonter en haut du classement.

---

## 4. Métriques de Performance et Audit

### Évaluation Globale (Test Set)
Nous avons privilégié le F1-Score comme métrique de référence pour équilibrer la précision (éviter les faux positifs) et le rappel (ne pas rater de bons candidats).

- Seuil de Tri Optimal : 0.590
- F1-Score (Invité) : 0.48
- Précision : 0.45 (Près d'un candidat sur deux suggéré est pertinent)
- Rappel : 0.50 (L'IA identifie 50% des meilleurs profils du dataset)
- Accuracy Globale : 0.72

### Audit d'Équité (Ethics First)
L'audit final montre une nette amélioration de l'équité par rapport aux versions initiales :
- Genre (Misandrie) : L'écart de rappel entre hommes et femmes a été réduit de 10.3% à 9.1%. Le modèle est devenu plus neutre.
- Âge (Biais Jeune) : Le modèle ne favorise plus les jeunes par simple défaut d'âge, mais les évalue sur leur mérite réel via le score de potentiel.
- Transparence : L'analyse SHAP confirme que le modèle ne repose plus sur des "proxies" de biais mais sur des compétences tangibles.

---

## 5. Analyse Visuelle

### Courbe ROC
L'AUC de 0.81 démontre une solide capacité de séparation des classes. Le modèle sait ordonner les candidats de manière logique.

![Courbe ROC](plots/roc_curve.png)

### Matrice de Confusion
La matrice confirme l'efficacité du filtrage des profils non pertinents (Vrais Négatifs), libérant du temps pour le recruteur.

![Matrice de Confusion](plots/confusion_matrix.png)

### Importance des Variables
Ce graphique montre que l'expérience et l'éducation restent les piliers, mais que le potentiel est désormais un levier de classement reconnu.

![Importance des Variables](plots/feature_importance.png)

---

## 6. Conclusion
Ce projet démontre qu'il est possible de concilier performance prédictive et éthique de recrutement. En stabilisant le modèle par la régularisation et en créant des variables de potentiel équitables, nous fournissons un outil de tri transparent et responsable. La prochaine étape cruciale sera l'acquisition de données supplémentaires (1000+ CV) pour affiner encore davantage la précision du système.
