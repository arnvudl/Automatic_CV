# 📋 Rapport d'Architecture : Pipeline d'Ingestion AUTOMATIC-CV

Ce document détaille la configuration du serveur, du réseau Docker et du workflow n8n permettant la réception automatisée de candidatures.

---

## 1. Infrastructure Serveur & Réseau 🌐

### Reverse Proxy & Sécurité
* **Serveur Web :** Nginx (installé sur l'hôte).
* **Domaine :** `https://n8n.lony.app`
* **SSL :** Certificat Let's Encrypt généré via Certbot.
* **Rôle :** Redirection du trafic HTTPS (port 443) vers le conteneur Docker n8n (port 5678) avec support des WebSockets pour l'interface temps réel.

### Architecture Docker
Deux services principaux tournent en isolation dans un réseau Docker commun :
1.  **`cv_n8n` :** Instance n8n pour l'orchestration.
2.  **`cv_api` :** Pipeline FastAPI (Python) pour la réception et le traitement.
    * *Note technique :* La communication entre n8n et l'API se fait via le nom de service Docker `http://api:8000`.

---

## 2. Configuration détaillée du Workflow n8n 🤖

Le workflow est composé de 5 nœuds chaînés pour transformer un email brut en une entrée structurée dans l'API.

### Nœud 1 : Déclencheur (Schedule)
* **Type :** Cron / Schedule.
* **Fréquence :** Toutes les 2 minutes.
* **Objectif :** Automatiser la vérification de la boîte mail.

### Nœud 2 : Récupération (HTTP Request)
* **Méthode :** `GET`
* **URL :** `https://api.testmail.app/api/json?apikey=[KEY]&namespace=73cn1`
* **Format de réponse :** `JSON`.
* **Donnée extraite :** Liste des emails reçus sous forme d'objets JSON.

### Nœud 3 : Filtrage (Filter)
* **Condition :** `{{ $json.emails[0].attachments }}` **Is Not Empty**.
* **Objectif :** Éliminer les emails publicitaires ou sans pièces jointes pour ne pas saturer le pipeline.

### Nœud 4 : Téléchargement (HTTP Request - Binary)
* **Méthode :** `GET`
* **URL :** `{{ $json.emails[0].attachments[0].downloadUrl }}`
* **Format de réponse :** `File` (Binaire).
* **Objectif :** Récupérer le contenu réel du fichier (CV) dans la mémoire de n8n.

### Nœud 5 : Transmission (HTTP Request - POST)
C'est le nœud critique qui fait le lien avec ton code Python.
* **Méthode :** `POST`
* **URL :** `http://api:8000/api/v1/candidates`
* **Type de contenu :** `Form-Data` (Multipart).
* **Paramètres envoyés :**
    * `file` (Binary) : Le fichier récupéré au Nœud 4.
    * `filename` (String) : Nom d'origine du CV.
    * `from` (String) : Adresse email de l'expéditeur (clé corrigée pour correspondre à l'alias FastAPI).

---

## 3. Résolution des incidents techniques 🛠️

| Problème | Cause | Solution |
| :--- | :--- | :--- |
| **Erreur de Cookie n8n** | Accès via IP vs Domaine. | Installation de Nginx et SSL pour accès via `https`. |
| **Domaine HS** | Traefik absent du Droplet. | Remplacement par Nginx configuré manuellement. |
| **Erreur 422 API** | Clé `from (address)` non reconnue. | Renommage de la clé en `from` dans n8n (match avec l'alias Form FastAPI). |
| **Espace Disque** | Images Docker orphelines. | Nettoyage via `docker image prune -f`. |

---

## 4. Statut Final 🚀

* **Interface n8n :** Opérationnelle sur `https://n8n.lony.app`.
* **Pipeline API :** Reçoit les fichiers et renvoie un `candidate_id` avec le statut `awaiting_review`.
* **Logs :** Visibles en temps réel via `docker logs cv_api -f`.

---

## 5. Évolution — Phase ML (Avril 2026)

Le pipeline n8n reste inchangé. La Phase 2 et 3 du projet ont été réalisées **hors n8n**, en local, sur des données synthétiques.

### Ce qui a été fait (indépendamment de n8n)

| Étape | Fichier | Résultat |
| :--- | :--- | :--- |
| Parsing + pseudonymisation | `pipeline_ml/parse_cv.py` | `data/processed/features.csv` + `identities.csv` |
| Génération pseudo-labels | `pipeline_ml/pseudo_labels.py` | Colonne `label` remplie (seuil score >= 12/16) |
| Entraînement ML | `pipeline_ml/train.py` | `models/model.pkl` (Random Forest, F1=0.837) |

### Connexion future n8n → ML

Quand le modèle sera validé, le nœud 5 (POST vers FastAPI) sera étendu :

```
Nœud 5 (actuel) : POST /api/v1/candidates  →  candidate_id + awaiting_review
         ↓
Nœud 6 (futur)  : POST /api/v1/score       →  score ML (0.0–1.0) + features extraites
```

Le scoring sera déclenché automatiquement après réception du CV, sans intervention humaine. La décision finale reste toujours humaine (AI Act).