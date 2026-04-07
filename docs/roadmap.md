### 📝 1. Parsing (Extraction des données)
Ton ami a rédigé un script robuste basé sur des expressions régulières (Regex) qui fonctionne parfaitement si les CV ont une structure homogène.

* **Conseil stratégique :** Garde ce script comme base. Les Regex sont extrêmement rapides et ne coûtent rien en puissance de calcul comparé à un modèle NLP lourd (comme un LLM). Si la structure de tes CV varie plus tard, tu pourras évoluer vers du NLP pur (spaCy ou Textract).
* **To-Do :**
    1.  Copier le fichier `extraction_cv.py` dans le dossier de ton API (sur DigitalOcean).
    2.  Modifier ton `main.py` (FastAPI) pour qu'il lise le fichier binaire reçu depuis n8n, le décode en texte (`.decode('utf-8')`), et l'envoie à la fonction `parse_cv`.

### 🕵️‍♂️ 2. Pseudonymisation (Gestion de l'identité)
Tu as parfaitement raison : la pseudonymisation est la seule voie viable en recrutement. Tu dois pouvoir recontacter le candidat si le modèle valide son profil.

* **Conseil stratégique :** Le modèle ML ne doit **jamais** voir le nom ou l'email. Séparer l'identité des compétences au niveau de la base de données.
* **To-Do :**
    1.  Dans FastAPI, générer un identifiant unique (`cv_id` = UUID).
    2.  Créer deux tables sur Supabase : 
        * `candidats_identites` : `cv_id`, `Nom`, `Email`, `Telephone`.
        * `candidats_features` : `cv_id`, `nb_jobs`, `years_experience`, `education_level`.

### 🛑 3. Critères Minimums (Hard Filters)
Un modèle de Machine Learning consomme des ressources et peut parfois halluciner. Il ne faut pas le solliciter pour des profils évidents à rejeter.

* **Conseil stratégique :** Implémente un filtre déterministe (règles métier) juste après le parsing et *avant* le Machine Learning. Si le candidat échoue ici, le pipeline s'arrête et le statut passe en "Rejet Auto".
* **To-Do :**
    1.  Écrire une fonction de validation simple dans FastAPI : `if data['years_experience'] < 2: return "Rejeté"` (exemple).
    2.  Si validé, passer les données au modèle ML.

### 🧠 4. Machine Learning (Prédiction)
Le modèle est effectivement une simple fonction mathématique (ex: Régression Logistique ou Random Forest) qui prend tes données parsées et sort un score (0 ou 1).

* **Conseil stratégique :** Entraîne ton modèle sur ton PC local en utilisant le fichier CSV généré par le script de ton ami. Exporte ensuite ce modèle sous forme de fichier binaire léger (format `.pkl` avec `joblib` ou `pickle`).
* **To-Do :**
    1.  Entraîner le modèle localement (Notebook Jupyter).
    2.  Exporter le modèle entraîné (`model.pkl`).
    3.  Importer `model.pkl` dans ton conteneur Docker FastAPI.
    4.  Faire un `model.predict([features_du_candidat])` dans ton `main.py`.

### ⚖️ 5. Audit des biais (Éthique & Conformité)
Le script de parsing extrait des données sensibles comme le `Gender`.

* **Conseil stratégique :** **Interdiction formelle** de fournir la variable `Gender` (ou l'âge, l'origine) au modèle de Machine Learning lors de l'entraînement ou de la prédiction. 
* **To-Do :**
    1.  Exclure `Gender` et `Age` du tableau de `features` envoyé au modèle.
    2.  Sauvegarder tout de même ces données dans Supabase à des fins de monitoring uniquement.
    3.  Créer une requête SQL ou un Dashboard Supabase pour vérifier le ratio d'acceptation : *(Nombre de Femmes acceptées / Nombre total de Femmes)* vs *(Nombre d'Hommes acceptés / Nombre total d'Hommes)* pour s'assurer que le modèle est neutre.