# Roadmap MLOps et OCR Avancé : MLflow, DVC et GLM-OCR

---

## État d'avancement

| Phase | Objectif | Statut |
|---|---|---|
| 1 | MLflow — tracking expérimentations et modèles | ✅ Terminé |
| 2 | Parsing multi-format + structuration LLM | ✅ Terminé |
| 3 | GLM-OCR — fallback PDF graphiques | 📅 Planifié |
| 4 | DVC — versionnage données et pipelines | 📅 Planifié |

---

## Périmètre du parsing et de l'OCR

Le parsing de CV opère dans **deux contextes distincts** qui doivent tous les deux être couverts :

| Contexte | Déclencheur | Chemin |
|---|---|---|
| **Entraînement** (batch) | Lancement manuel de `p01_parse.py` | `data/raw/` → `features.csv` + `identities.csv` |
| **Inférence live** | Email reçu avec CV en pièce jointe (n8n → FastAPI) | n8n télécharge le fichier → POST multipart `/score` → réponse JSON avec score |

L'OCR (Phase 3) et le routage multi-format (Phase 2 ✅) doivent fonctionner dans les **deux contextes**. Un CV PDF graphique reçu par email doit être lisible autant qu'un CV d'entraînement.

---

## Phase 1 — MLflow ✅

**Terminé.** Tracking intégré dans `p04_train.py` et `p06_audit.py`.

### Ce qui a été fait

- `p04_train.py` : log des hyperparamètres (`C`, `l1_ratio`, `solver`, seuils adulte/junior), métriques (`auc_roc_cv5`, `auc_roc_test`, `f1_test`), artefact `evaluation.txt` et modèle sklearn signé. Sauvegarde du `run_id` dans `models/mlflow_run_id.txt`.
- `p06_audit.py` : reprend le même run MLflow via `mlflow_run_id.txt`, log des métriques de fairness (`recall_female`, `recall_male`, `recall_gap_male_female`, `recall_age_*`), SHAP par feature (`shap_<feature>`), artefact `audit.txt`.
- `mlruns/` et `mlflow.db` exclus du git (`.gitignore`).

### Utilisation

```bash
python -m pipeline_ml.core.p04_train
python -m pipeline_ml.core.p06_audit
mlflow ui   # → http://localhost:5000
```

### À faire (améliorations futures)

- **Model Registry** : promouvoir un run en `Production` via l'UI ou l'API MLflow pour savoir exactement quel `.pkl` est déployé.
- Ajouter les courbes ROC et matrices de confusion comme artefacts image.

---

## Phase 2 — Parsing multi-format + LLM ✅

**Terminé.** Routage intelligent dans `p01_parse.py`, couvrant entraînement et inférence live.

### Ce qui a été fait

**Pipeline entraînement (`p01_parse.py`) :**
- `extract_text()` : router par extension — `.txt` natif, `.pdf` via pdfplumber, `.docx` via python-docx.
- `main()` : glob `*.txt + *.pdf + *.docx` dans `data/raw/`.
- Routage automatique : `.txt` + mode `--parser regex` → `parse_cv()` rapide en parallèle ; PDF/DOCX → extraction texte puis `parse_cv_llm()` (Groq).
- Warning explicite si pdfplumber ne sort pas assez de texte → signal OCR requis (Phase 3).

**Inférence live (FastAPI `/score`) :**
- L'endpoint reçoit le fichier en multipart (envoyé par n8n depuis la pièce jointe email).
- Il doit appeler `extract_text()` puis `parse_cv_llm()` — **à vérifier que `api/main.py` utilise bien ces fonctions** et non une ancienne version regex-only.

**Structuration LLM (Groq `llama-3.3-70b-versatile`) :**
- Prompt JSON schema robuste : extrait nom, email, téléphone, formation, expériences, compétences, langues, certifications.
- Fallback regex si le LLM échoue sur un fichier `.txt` structuré.

### Packages ajoutés

```
mlflow, pdfplumber, python-docx
```

---

## Phase 3 — GLM-OCR (fallback PDF graphiques) 📅

**Non démarré.**

### Problème ciblé

pdfplumber échoue sur les PDF sans couche texte : CV en colonnes complexes, CV Canva/infographiques, scans de documents papier. Ces cas génèrent un warning et sont actuellement ignorés.

### Périmètre — les deux contextes

- **Entraînement** : CVs PDF dans `data/raw/` qui reviennent vides après pdfplumber.
- **Inférence live** : CV reçu par email en PDF graphique → doit quand même être scoré, pas silencieusement ignoré.

### Architecture cible

```
fichier .pdf
   │
   ├─ pdfplumber → texte suffisant (> 100 chars) ?
   │     ├─ Oui → parse_cv_llm(texte)
   │     └─ Non → GLM-OCR (vision-language)
   │               └─ texte OCR → parse_cv_llm(texte)
```

GLM-OCR : [github.com/zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR) — modèle vision-language qui comprend la structure visuelle d'un document complexe.

### Plan d'action

1. Évaluer GLM-OCR sur un échantillon de CVs PDF complexes de `data/raw/`.
2. Intégrer dans `_extract_text_pdf()` en fallback (remplace le `return ""` actuel).
3. Même logique dans l'endpoint FastAPI `/score` pour l'inférence live.
4. Tester sur des CVs Canva reçus par email.

---

## Phase 4 — DVC (industrialisation) 📅

**Non démarré.**

### Objectifs

- **Reproductibilité** : associer chaque commit Git à une version exacte des données (`data/raw/`, `data/processed/`) et des modèles (`models/`).
- **Stockage distant sécurisé** : pousser les CVs (données sensibles) vers un remote cloud (S3, Google Drive) — seules des métadonnées légères (`.dvc`) restent dans Git.
- **Pipeline DAG** : définir `p01 → p02 → p04 → p06` dans un `dvc.yaml`. DVC sait quelles étapes relancer si seul le code d'audit change.

### Plan d'action

1. `dvc init` dans le projet.
2. Créer `dvc.yaml` pour le pipeline complet.
3. Configurer un remote (ex: `dvc remote add -d myremote s3://bucket/cv-intelligence`).
4. `dvc add data/raw/ data/processed/ models/` → génère les `.dvc` à committer.

---

*Dernière mise à jour : Mai 2026*
