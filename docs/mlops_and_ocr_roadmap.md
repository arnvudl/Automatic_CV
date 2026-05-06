# Roadmap MLOps et OCR Avancé : Intégration de MLflow, DVC et GLM-OCR

Dans le cadre de l'amélioration continue du projet CV-Intelligence et de la transition vers des pratiques MLOps robustes, ce document détaille l'intégration prévue de nouveaux outils majeurs : MLflow, DVC, GLM-OCR et un LLM via API.

---

## 1. Suivi des Expérimentations et des Modèles avec MLflow

Actuellement, les métriques d'évaluation et les rapports d'équité (Fairness) sont générés sous forme de fichiers texte (ex: `reports/RAPPORT_FINAL_ML.md`, `reports/evaluation.txt`). Bien que fonctionnel, ce système manque de traçabilité historique et ne permet pas de comparer facilement deux itérations du modèle.

### Objectifs de l'intégration de MLflow
- **Tracking Automatique** : Enregistrer automatiquement les hyperparamètres (ex: paramètre de régularisation `C`, seuils de décision `threshold` et `threshold_junior`) à chaque exécution de l'entraînement (`p04_train.py`, `p05_tune.py`).
- **Journalisation des Métriques** : Sauvegarder les métriques globales (AUC-ROC, F1-Score) et surtout les métriques d'équité (Recall par âge, genre, pays) générées lors de l'audit (`p06_audit.py`).
- **Gestion des Artefacts** : Lier de façon permanente les graphiques générés (matrices de confusion, SHAP, courbes ROC) au modèle exact qui les a produits, sans écraser les anciens fichiers.
- **Registre de Modèles (Model Registry)** : Versionner les fichiers `.pkl` (modèle, scaler, seuils) pour savoir exactement quel modèle est actuellement déployé en production.

---

## 2. Versionnage des Données et des Pipelines avec DVC (Data Version Control)

Les données brutes (CVs), les données traitées (`features.csv`, `identities.csv`), et les modèles ne sont pas suivis par Git en raison de leur taille et de leur nature sensible. DVC permettra de versionner ces éléments de manière sécurisée et reproductible.

### Objectifs de l'intégration de DVC
- **Reproductibilité** : Associer chaque commit Git à une version exacte des données et du modèle. Il sera ainsi possible de revenir à l'état exact du dataset lors de la création de la version "V5" du modèle.
- **Stockage Distant Sécurisé** : Pousser les données lourdes ou sensibles vers un stockage cloud (ex: S3, Google Drive) tout en ne conservant que des métadonnées légères dans le dépôt Git (via des fichiers `.dvc`).
- **Pipelines DAG** : Définir le flux de travail complet (`p01_parse.py` -> `p02_features.py` -> `p04_train.py` -> `p06_audit.py`) sous forme de graphe (DAG). Si seul le code de l'audit est modifié, DVC saura qu'il est inutile de relancer le parsing et l'extraction des features.

---

## 3. Extraction Avancée et Tri par LLM (Parsing intelligent)

Le module de parsing actuel (`p01_parse.py`) repose principalement sur des expressions régulières qui peuvent s'avérer fragiles face à la diversité des mises en page des CVs.

### Architecture de Parsing Cible
L'objectif est de mettre en place un **routage intelligent** selon le format du fichier, suivi d'une étape de structuration par un LLM :

1. **Extraction brute optimisée (selon format) :**
   - `.txt` : Lecture native Python (Rapide et sans erreur).
   - `.docx` : Extraction via librairie (ex: `python-docx`) (Rapide et préserve la structure de base).
   - `.pdf` :
     - Tentative d'extraction native de texte (ex: `pdfplumber`).
     - En cas d'échec ou de CV trop graphique (colonnes complexes, infographies), recours à **GLM-OCR** ([dépôt GitHub](https://github.com/zai-org/GLM-OCR)) : un modèle vision-language ultra-puissant capable de comprendre la structure visuelle d'un document complexe.

2. **Structuration et tri par LLM via API :**
   - Peu importe la méthode d'extraction brute, le texte résultant (souvent bruité ou désorganisé) sera envoyé à un **LLM via API** (ex: OpenAI, Anthropic, Mistral).
   - **Rôle du LLM** : Analyser le texte brut et extraire *uniquement* les informations pertinentes au format JSON (nom, e-mail, téléphone, compétences techniques, nombre d'années d'expérience, etc.).
   - Le LLM agit comme un filtre intelligent qui s'affranchit des variations de mise en page et des formulations propres à chaque candidat.

---

## Plan d'Action Proposé

1. **Phase 1 : MLflow (Quick Win)**
   - Installer `mlflow`.
   - Modifier `p04_train.py` et `p06_audit.py` pour logger les paramètres, les métriques d'équité et sauvegarder les graphiques SHAP/ROC en tant qu'artefacts MLflow.
2. **Phase 2 : Extraction structurée via LLM**
   - Refactoriser `p01_parse.py` pour intégrer un appel API vers un LLM.
   - Créer un prompt d'extraction (JSON schema) robuste pour trier les informations du CV brut.
3. **Phase 3 : Intégration de GLM-OCR**
   - Évaluer GLM-OCR sur un sous-ensemble de CVs PDF complexes où l'extraction de texte classique échoue.
   - L'intégrer en *fallback* (plan B) dans le pipeline de parsing pour les PDF graphiques.
4. **Phase 4 : DVC (Industrialisation)**
   - Initialiser DVC dans le projet.
   - Créer un fichier `dvc.yaml` pour définir le pipeline complet de préparation des données et d'entraînement.
   - Configurer un remote DVC pour le stockage sécurisé des datasets bruts et annotés.