# ML Training Checklist — CV-Intelligence
## Du pré-traitement au modèle production-ready

*Guide complet des étapes, méthodes, et contrôles requis pour la Phase 3 et 4 (Entraînement + Audit + Conformité)*

---

## 📋 Table des matières

1. [Préparation des données](#1-préparation-des-données)
2. [Split train/val/test](#2-split-trainvaltest)
3. [Feature engineering et scaling](#3-feature-engineering-et-scaling)
4. [Entraînement du modèle](#4-entraînement-du-modèle)
5. [Évaluation globale](#5-évaluation-globale)
6. [Audit biais et fairness](#6-audit-biais-et-fairness)
7. [Explicabilité et SHAP](#7-explicabilité-et-shap)
8. [Validation humaine et prise de décision](#8-validation-humaine-et-prise-de-décision)
9. [Déploiement et monitoring](#9-déploiement-et-monitoring)
10. [Documentation et traçabilité](#10-documentation-et-traçabilité)

---

## 1. Préparation des données

### 1.1 Charger et valider

**Objectif** : Assurer que `data/processed/features.csv` est complet et conforme.

#### Checklist

- ☐ **Charge du CSV** → vérifier que toutes les colonnes attendues sont présentes :
  - `cv_id` (UUID, clé primaire)
  - `nb_jobs` (entier)
  - `years_experience` (float)
  - `education_level` (catégorique : Bac, Licence, Master, Doctorat)
  - `skills` (liste de compétences requises, séparées par `;` ou stockées comme JSON)
  - `languages` (ex. anglais, français, allemand)
  - Target variable (label) : `hired` (booléen 0/1) ou `recommendation` (0-1 score)
  
- ☐ **Absence d'identifiants** : vérifier que `nom`, `prénom`, `email`, `téléphone`, `age`, `gender` n'existent **pas** dans ce fichier
  
- ☐ **Taille du dataset** : combien de lignes avez-vous ? (Recommandé : min 100-200 CV pour un MVP)
  
- ☐ **Valeurs manquantes** :
  ```python
  # Identifier les NaN
  missing = df.isnull().sum()
  print(missing)
  # Stratégie : 
  # - years_experience NaN → imputer à 0
  # - skills manquant → imputer à "" (liste vide)
  # - education_level NaN → mode (+ attention section 1.2)
  ```
  
- ☐ **Valeurs aberrantes** :
  - `nb_jobs` > 50 ? 
  - `years_experience` > 70 ?
  - Education level invalide ?
  → Documenter, décider : garder, imputer, ou isoler pour révision humaine
  
- ☐ **Distribution du target** :
  ```python
  print(df['hired'].value_counts())
  print(df['hired'].value_counts(normalize=True))
  # Alerte : si déséquilibre > 80/20 ou < 20/80, noter pour l'étape 4
  ```

### 1.2 Imbalance dataset & stratification

**Si le target est déséquilibré (ex. 85% hired, 15% rejected)** :

- ☐ **Ne pas balancer arbitrairement** avant split (réduirait l'info réelle)
- ☐ **Utiliser stratification au split** (voir section 2) pour garder les proportions dans train/val/test
- ☐ **Évaluer les impacts** :
  - En production, qu'attend-on ? (si 85% des CV sont "bon fit", c'est normal)
  - La métrique pertinente n'est pas l'accuracy mais le recall/precision par classe

### 1.3 Anomalies métier

Avant de continuer, valider **logiquement** :

- ☐ `years_experience >= 0` pour tous les profils
- ☐ Si `education_level` = "Bac" et `years_experience` = 30, pas d'incohérence métier ?
- ☐ Les `skills` correspondent-elles à la description du poste ?
- ☐ Y a-t-il des patterns bizarres (ex. tous les rejected ont <2 ans d'expérience) ?

**Documenter** tous les choix d'imputation / suppression / anomalies.

---

## 2. Split train/val/test

### 2.1 Proportions recommandées

- **Train** : 60-70% (données pour apprendre)
- **Validation** : 15-20% (données pour régler hyperparamètres pendant entraînement)
- **Test** : 15-20% (données **jamais vues** pendant entraînement, pour évaluation finale)

```python
from sklearn.model_selection import train_test_split

# Étape 1 : Séparer 80% train+val, 20% test
train_val, test = train_test_split(
    df, 
    test_size=0.20,
    random_state=42,  # Reproductibilité
    stratify=df['hired']  # Garder les proportions
)

# Étape 2 : Séparer train+val en train/val
train, val = train_test_split(
    train_val,
    test_size=0.20,  # 20% de 80% = 16% du total
    random_state=42,
    stratify=train_val['hired']
)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
# Exemple output : Train: 320, Val: 80, Test: 100
```

### 2.2 Checklist

- ☐ **Random state fixé** à 42 (ou autre, mais constant) pour reproductibilité
- ☐ **Stratification activée** sur le target pour datasets déséquilibrés
- ☐ **Pas de data leakage** : le test set n'a été vu par AUCUN étape de preprocessing
- ☐ **Sauvegarder les indices** des 3 sets pour pouvoir revenir à l'original si besoin

```python
# Sauvegarder
train.to_csv('data/processed/train.csv', index=False)
val.to_csv('data/processed/val.csv', index=False)
test.to_csv('data/processed/test.csv', index=False)
```

---

## 3. Feature engineering et scaling

### 3.1 Feature engineering

**Créer des features pertinentes à partir des colonnes brutes.**

#### Colonnes numériques

- ☐ `years_experience` : potentiellement créer des bins (0-2 ans, 2-5, 5-10, 10+)
  ```python
  df['exp_category'] = pd.cut(df['years_experience'], 
                               bins=[0, 2, 5, 10, 100],
                               labels=['junior', 'intermediate', 'senior', 'expert'])
  ```
- ☐ `nb_jobs` : garder tel quel ou créer ratio `job_changes_per_year = nb_jobs / years_experience` ?

#### Colonnes catégoriques

- ☐ `education_level` : one-hot encoding (créer colonnes binaires)
  ```python
  edu_dummies = pd.get_dummies(df['education_level'], prefix='edu', drop_first=True)
  # Crée : edu_licence, edu_master, edu_doctorat
  ```

#### Colonnes texte / listes

- ☐ `skills` : plusieurs stratégies
  - Option A : **Count** : combien de compétences requises détectées dans le CV ?
    ```python
    required_skills = set(['Python', 'SQL', 'ML'])  # À définir par rôle
    df['skill_match_count'] = df['skills'].apply(
        lambda s: len([sk for sk in s.split(';') if sk in required_skills])
    )
    ```
  - Option B : **TF-IDF** ou **embeddings** si prise en charge future (avancé)
  
- ☐ `languages` : binary features
  ```python
  df['speaks_english'] = df['languages'].str.contains('anglais', case=False).astype(int)
  df['speaks_french'] = df['languages'].str.contains('français', case=False).astype(int)
  ```

### 3.2 Sélection des features pour le modèle

**Rappel RGPD** : le modèle ne reçoit **jamais** `age`, `gender`, `nationalité`.

**Features finales recommandées** :
- `years_experience`
- `exp_category` (si créé)
- `nb_jobs`
- `job_changes_per_year` (si créé)
- `education_level` (après one-hot : `edu_licence`, `edu_master`, etc.)
- `skill_match_count` (ou autre métrique de matching)
- `speaks_english`, `speaks_french`

```python
# Feature matrix (sans target)
feature_cols = [
    'years_experience', 'nb_jobs', 'job_changes_per_year',
    'edu_licence', 'edu_master', 'edu_doctorat',
    'skill_match_count', 'speaks_english', 'speaks_french'
]
X_train = train[feature_cols]
y_train = train['hired']
```

### 3.3 Scaling (normalisation)

**Obligatoire pour** : Régression Logistique, SVM, KNN, Neural Networks  
**Optionnel pour** : Random Forest, XGBoost (mais peut aider)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# ⚠️ IMPORTANT : FIT uniquement sur train, TRANSFORM sur val et test
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)  # Pas de fit !
X_test_scaled = scaler.transform(X_test)  # Pas de fit !

# Sauvegarder le scaler pour la production
import joblib
joblib.dump(scaler, 'models/scaler.pkl')
```

### 3.4 Checklist features

- ☐ Aucune colonne sensible (`age`, `gender`) ne figure dans X_train/X_val/X_test
- ☐ Scaler fitté uniquement sur train
- ☐ Même scaler appliqué à val et test (pas de refit)
- ☐ Scaler sauvegardé pour la production

---

## 4. Entraînement du modèle

### 4.1 Choix d'algorithmes

Selon votre roadmap, trois niveaux de complexité :

#### Niveau 1 : Régression Logistique (MVP rapide)

**Avantages** : rapide, interprétable, baseline solide  
**Inconvénients** : hypothèse de séparabilité linéaire

```python
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'  # Si déséquilibre classe
)

model_lr.fit(X_train_scaled, y_train)
```

#### Niveau 2 : Random Forest (bon compromis)

**Avantages** : robuste, gère non-linéarité, naturellement interprétable (feature importance)  
**Inconvénients** : moins explicable que LR au niveau prédiction individuelle

```python
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1  # Parallélisation
)

model_rf.fit(X_train, y_train)  # Pas besoin de scaling
```

#### Niveau 3 : XGBoost (haute performance)

**Avantages** : très bon score, gère déséquilibre, scalable  
**Inconvénients** : boîte noire, risque overfitting, hyperparamètres sensibles

```python
import xgboost as xgb

model_xgb = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=1,  # À ajuster selon déséquilibre
    n_jobs=-1
)

model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
```

### 4.2 Hyperparamètres

**Stratégie** : grid search ou random search sur validation set

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [5, 10, 15],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='f1',  # À adapter (voir section 5)
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")
```

### 4.3 Checklist entraînement

- ☐ Modèle choisi documenté + justification
- ☐ Hyperparamètres documentés
- ☐ Pas de fuite (eval_set ne contient que val, pas test)
- ☐ Modèle sauvegardé :
  ```python
  joblib.dump(best_model, 'models/model.pkl')
  ```

---

## 5. Évaluation globale

### 5.1 Choisir les bonnes métriques

**Ne pas vous arrêter à l'accuracy.**

Pour un recrutement (classification binaire déséquilibrée) :

| Métrique | Formule | Quand l'utiliser | Pièges |
|----------|---------|------------------|-------|
| **Accuracy** | (TP + TN) / Total | Données équilibrées | Trompeuse si déséquilibre |
| **Precision** | TP / (TP + FP) | On veut peu de faux positifs | Ignore les faux négatifs |
| **Recall** (Sensibilité) | TP / (TP + FN) | On veut peu de faux négatifs | Peut accepter plus FP |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Équilibre P et R | Symétrique (pas toujours bon) |
| **ROC-AUC** | Aire sous la courbe ROC | Évaluer seuil de décision | Ignore les vrais négatifs |
| **PR-AUC** | Aire sous courbe Precision-Recall | Déséquilibre marqué | Courbe moins intuitive |

**Pour le recrutement, je recommande** :
- Calcul sur test set
- Metric principale : **F1-score** (ou **Precision/Recall weighted** si déséquilibre)
- Secondaires : Confusion matrix, ROC-AUC

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report
)

y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print("=" * 50)
print("ÉVALUATION TEST SET")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

**Résultat cible** :
- F1 > 0.70 (bon)
- Pas de recall < 0.60 (trop de faux négatifs)
- Precision > 0.65 (limiter faux positifs)

### 5.2 Validation croisée (optionnel mais recommandé)

```python
from sklearn.model_selection import cross_validate

scores = cross_validate(
    best_model,
    X_train_scaled,
    y_train,
    cv=5,  # 5-fold
    scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
    return_train_score=False
)

print("Cross-validation results:")
for metric, values in scores.items():
    print(f"{metric}: {values.mean():.3f} ± {values.std():.3f}")
```

### 5.3 Courbes d'apprentissage

Détecter overfitting vs underfitting :

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring='f1',
    train_sizes=np.linspace(0.1, 1, 10),
    n_jobs=-1
)

plt.plot(train_sizes, train_scores.mean(axis=1), 'b-', label='Train F1')
plt.plot(train_sizes, val_scores.mean(axis=1), 'r-', label='Val F1')
plt.xlabel('Training size')
plt.ylabel('F1-Score')
plt.legend()
plt.savefig('reports/learning_curve.png')
```

**Interprétation** :
- Si train F1 haut et val F1 bas → **overfitting** (réduire complexité)
- Si les deux bas → **underfitting** (augmenter complexité)

### 5.4 Checklist évaluation

- ☐ Prédictions faites sur test set (jamais vu)
- ☐ Metrics calculées : Accuracy, Precision, Recall, F1, ROC-AUC
- ☐ Confusion matrix inspectée
- ☐ Pas d'overfitting flagrant
- ☐ Performance documentée dans rapport

---

## 6. Audit biais et fairness

**CRITIQUE pour RGPD et AI Act.**

### 6.1 Données de biais (stockées séparément)

Vous avez gardé `age`, `gender` dans un fichier séparé pour le monitoring.

```python
# Charger les métadonnées sensibles (test set uniquement)
metadata_test = pd.read_csv('data/sensitive/metadata_test.csv')
# Colonnes : cv_id, age, gender

# Mapper aux prédictions
test_with_meta = test.copy()
test_with_meta['predicted_label'] = y_pred
test_with_meta['predicted_proba'] = y_pred_proba
test_with_meta = test_with_meta.merge(metadata_test, on='cv_id')
```

### 6.2 Métriques de fairness

#### Disparate Impact Ratio (DI)

Pour chaque groupe (genre, âge), calculer :

```python
# Par genre
for gender in ['M', 'F']:
    subset = test_with_meta[test_with_meta['gender'] == gender]
    accept_rate = subset['predicted_label'].mean()
    print(f"Accept rate ({gender}): {accept_rate:.1%}")

# DI = acceptance_rate_female / acceptance_rate_male
# Règle 80% : DI >= 0.80 est acceptable (légalement)
# Cible : DI > 0.95 (très bon)
```

#### Demographic Parity

```python
from fairlearn.metrics import demographic_parity_difference

dpd = demographic_parity_difference(
    y_test,
    y_pred,
    sensitive_features=test_with_meta['gender']
)
print(f"Demographic Parity Difference: {dpd:.3f}")
# Objectif : proche de 0
```

#### Equalized Odds

Même acceptance rate ET même error rate par groupe :

```python
from fairlearn.metrics import equalized_odds_difference

eod = equalized_odds_difference(
    y_test,
    y_pred,
    sensitive_features=test_with_meta['gender']
)
print(f"Equalized Odds Difference: {eod:.3f}")
# Objectif : proche de 0
```

### 6.3 Analyse par âge

```python
# Créer groupes d'âge
test_with_meta['age_group'] = pd.cut(
    test_with_meta['age'],
    bins=[0, 30, 40, 50, 100],
    labels=['18-30', '30-40', '40-50', '50+']
)

for age_group in ['18-30', '30-40', '40-50', '50+']:
    subset = test_with_meta[test_with_meta['age_group'] == age_group]
    if len(subset) > 0:
        accept_rate = subset['predicted_label'].mean()
        n = len(subset)
        print(f"{age_group}: {accept_rate:.1%} (n={n})")
```

**Alerte** : Si une classe d'âge a 0% d'acceptation, investiguer !

### 6.4 Matrice de confusion par groupe

```python
from sklearn.metrics import confusion_matrix

for gender in ['M', 'F']:
    subset_idx = test_with_meta['gender'] == gender
    y_test_sub = y_test[subset_idx]
    y_pred_sub = y_pred[subset_idx]
    
    cm = confusion_matrix(y_test_sub, y_pred_sub)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n{gender}:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
```

### 6.5 Thresholds d'alerte

Documenter et décider des seuils d'action :

| Situation | Seuil | Action |
|-----------|-------|--------|
| DI < 0.80 | Légal | ⚠️ Risque légal → Revoir modèle ou hard filters |
| DI 0.80-0.95 | Marginal | 🟡 Acceptable mais surveiller |
| DI > 0.95 | Bon | ✅ Valider en production |
| Accept rate diff > 20% entre âges | Préoccupant | 🟡 Investiguer si corrélé aux données |

### 6.6 Checklist audit biais

- ☐ Données sensibles chargées (age, gender) **uniquement pour monitoring**
- ☐ Disparate Impact Ratio calculé par genre
- ☐ Demographic Parity Difference < 0.10
- ☐ Equalized Odds Difference < 0.10
- ☐ Analyse par âge : pas de groupe à 0% ou 100%
- ☐ Confusion matrix par genre : pas de forte asymétrie
- ☐ Thresholds d'alerte documentés et décision prise
- ☐ **Rapport biais écrit et validé par une personne non-technique** (RH, légal)

---

## 7. Explicabilité et SHAP

### 7.1 Feature importance (global)

#### Random Forest / XGBoost

```python
import matplotlib.pyplot as plt

# Feature importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importance')
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_cols[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig('reports/feature_importance.png')

print("Top 5 features:")
for i in range(5):
    print(f"  {feature_cols[indices[i]]}: {importances[indices[i]]:.3f}")
```

#### Régression Logistique

```python
# Coefficients
coeffs = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': best_model.coef_[0],
    'abs_coeff': np.abs(best_model.coef_[0])
})
coeffs = coeffs.sort_values('abs_coeff', ascending=False)

print("Top features:")
print(coeffs.head(10))
```

**Attention RGPD** : Si `age` ou `gender` était accidentellement dans les features, vous verriez ici une importance élevée. C'est une alerte ! ⚠️

### 7.2 SHAP (Explainability par prédiction)

SHAP montre **pourquoi** le modèle a décidé pour un candidat spécifique.

```python
import shap

# Créer explainer
explainer = shap.TreeExplainer(best_model)  # Pour RF/XGB
# ou shap.LinearExplainer(best_model, X_train_scaled) pour LR

# Calculer SHAP values sur test set
shap_values = explainer.shap_values(X_test)

# Force plot pour une prédiction
shap.force_plot(
    explainer.expected_value,
    shap_values[0],  # Première prédiction du test set
    X_test[0],
    feature_names=feature_cols
)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_cols)
plt.savefig('reports/shap_summary.png')

# Dependence plot pour une feature clé
shap.dependence_plot('years_experience', shap_values, X_test, feature_names=feature_cols)
plt.savefig('reports/shap_dependence.png')
```

**Cas d'usage** : Quand un candidat conteste un rejet, le recruteur peut montrer la décomposition SHAP.

### 7.3 Checklist explicabilité

- ☐ Feature importance calculée et visualisée
- ☐ Top 5 features documentées et font sens métier
- ☐ **Aucune feature sensible (`age`, `gender`) dans top features**
- ☐ SHAP explainer entraîné
- ☐ Force plot testé pour quelques cas
- ☐ Explications documentées pour le dashboard RH

---

## 8. Validation humaine et prise de décision

### 8.1 Échantillon de révision manuelle

Avant de déployer, faire une **revue manuelle d'un sous-ensemble** du test set :

```python
# Sélectionner un échantillon stratifié
sample_size = 30  # 10-15% du test set
sample_idx = np.random.choice(len(y_test), sample_size, replace=False)

review_set = test.iloc[sample_idx].copy()
review_set['model_pred'] = y_pred[sample_idx]
review_set['model_proba'] = y_pred_proba[sample_idx]

review_set.to_csv('review/manual_review_sample.csv', index=False)
```

**Faire réviser par** : RH manager + recruiter + (optionnel) personne du métier

**Checklist de révision** :
- ☐ Pour chaque candidat, le recruteur évalue : accord ou désaccord avec le modèle ?
- ☐ Si désaccord → Pourquoi ? (Feature manquante ? Soft skills non capturées ?)
- ☐ Documenter les patterns de désaccord
- ☐ Taux d'accord attendu : >75% (sinon, retravailler les features)

### 8.2 Seuil de décision

Le modèle produit une probabilité (0-1). À partir de quel seuil classer en "hired" ?

**Default** : seuil = 0.5  
**Optimisé** : choisir le seuil selon votre tolérance à FP vs FN

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
plt.plot(thresholds, recall[:-1], 'r-', label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.savefig('reports/precision_recall_threshold.png')

# Choisir seuil
# Ex : threshold = 0.45 → meilleur F1
# Ex : threshold = 0.55 → plus conservateur (moins faux positifs)
threshold_choice = 0.50
print(f"Chosen threshold: {threshold_choice}")
```

### 8.3 Règles de hard filter vs soft score

**Hard filter** (déterministe, appliqué **avant** ML) :
- Expérience < 2 ans → rejected (pas de modèle)
- Aucune compétence requise → rejected (pas de modèle)

**Soft score** (modèle, produit probabilité) :
- prob > 0.60 → recommended for hiring
- 0.40 < prob < 0.60 → borderline, revue recommandée
- prob < 0.40 → not recommended

```python
def make_recommendation(prob, experience, skill_count):
    """
    Décision finale : hard filter + model score
    """
    # Hard filters
    if experience < 2:
        return 'REJECTED (hard filter: <2 years exp)', prob
    if skill_count == 0:
        return 'REJECTED (hard filter: no required skills)', prob
    
    # Model score
    if prob > 0.60:
        return 'RECOMMENDED', prob
    elif prob > 0.40:
        return 'BORDERLINE (needs review)', prob
    else:
        return 'NOT RECOMMENDED', prob

# Appliquer
test_with_recs = test.copy()
test_with_recs[['recommendation', 'model_proba']] = test_with_recs.apply(
    lambda row: make_recommendation(
        y_pred_proba[row.name],
        row['years_experience'],
        row['skill_match_count']
    ),
    axis=1,
    result_type='expand'
)

print(test_with_recs[['recommendation']].value_counts())
```

### 8.4 Checklist prise de décision

- ☐ Échantillon manuel révisionné et accord > 75%
- ☐ Seuil de probabilité documenté et justifié
- ☐ Catégories de recommandation définies (RECOMMENDED, BORDERLINE, NOT RECOMMENDED)
- ☐ Hard filters appliqués **avant** modèle
- ☐ Documentation claire : "Comment le modèle recommande-t-il ?"

---

## 9. Déploiement et monitoring

### 9.1 Packager le modèle pour production

```python
# Sauvegarder tous les artefacts
import joblib
import json

artifacts = {
    'model': best_model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'threshold': 0.50,
    'explainer': explainer  # SHAP
}

for name, obj in artifacts.items():
    joblib.dump(obj, f'models/{name}.pkl')

# Config JSON
config = {
    'model_type': 'RandomForest',
    'features': feature_cols,
    'threshold': 0.50,
    'version': '1.0',
    'trained_date': '2026-04-10',
    'test_f1': 0.78,
    'test_precision': 0.80,
    'test_recall': 0.76,
    'disparate_impact_ratio': 0.94
}

with open('models/config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

### 9.2 API FastAPI : intégrer le modèle

```python
# cv_api/main.py

from fastapi import FastAPI, UploadFile, File
import joblib
import json

app = FastAPI()

# Charger le modèle au démarrage
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')
explainer = joblib.load('models/explainer.pkl')

with open('models/config.json') as f:
    config = json.load(f)

@app.post("/predict")
async def predict(cv_id: str, features_dict: dict):
    """
    Input: {'years_experience': 5, 'nb_jobs': 3, ...}
    Output: {'recommendation': 'RECOMMENDED', 'probability': 0.72, 'explanation': {...}}
    """
    
    # Construire X
    X = [[features_dict.get(col, 0) for col in config['features']]]
    
    # Prédire
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]
    
    # SHAP explanation
    shap_vals = explainer.shap_values(X_scaled)[1]
    
    # Décision
    if prob > config['threshold']:
        recommendation = 'RECOMMENDED'
    else:
        recommendation = 'NOT RECOMMENDED'
    
    return {
        'cv_id': cv_id,
        'recommendation': recommendation,
        'probability': float(prob),
        'shap_explanation': shap_vals.tolist()
    }
```

### 9.3 Monitoring en production

**Exigence AI Act** : vérifier que le modèle ne dérive pas.

```python
# Chaque mois, collecter des métriques
def monitor_model(new_predictions, ground_truth_sample):
    """
    new_predictions: prédictions du mois dernier
    ground_truth_sample: feedback RH (acceptance/rejection confirmée)
    """
    
    # Récalculer F1
    f1 = f1_score(ground_truth_sample, new_predictions)
    
    # Vérifier DI par genre (si métadonnées disponibles)
    # ...
    
    # Alerte si F1 < 0.70
    if f1 < 0.70:
        print("⚠️ Model F1 dropped below 0.70. Retraining recommended.")
    
    return f1

# Stocker en base de données ou fichier log
```

### 9.4 Checklist déploiement

- ☐ Modèle + scaler + explainer packagés
- ☐ Config.json créé avec metrics et version
- ☐ API FastAPI intégrée
- ☐ Test end-to-end : requête API → prédiction + explication
- ☐ Monitoring setup : logs des prédictions en production
- ☐ Alerte mise en place si F1 < 0.70

---

## 10. Documentation et traçabilité

### 10.1 Rapport ML final (pour légal + RH)

Créer un document PDF/Word contenant :

```
1. RÉSUMÉ EXÉCUTIF
   - Objectif : scorer CV pour recrutement
   - Modèle : Random Forest, 100 estimators
   - Performance : F1=0.78, Precision=0.80, Recall=0.76
   - Status : ✅ Prêt à production

2. DONNÉES
   - Source : CVs reçus via n8n
   - Taille : 400 CVs (60% train, 20% val, 20% test)
   - Features : years_experience, nb_jobs, education_level, skills, languages
   - Données exclues (RGPD) : age, gender, name, email (jamais utilisées en ML)

3. MÉTHODOLOGIE
   - Train/val/test stratifié
   - Scaling : StandardScaler fitté sur train
   - Hyperparamètres : grid search sur validation set

4. RÉSULTATS
   - Accuracy: 0.75
   - Precision: 0.80 (peu de faux positifs)
   - Recall: 0.76 (peu de faux négatifs)
   - F1-Score: 0.78 (équilibre)
   - ROC-AUC: 0.84

5. AUDIT BIAIS
   - Disparate Impact Ratio (genre) : 0.94 ✅ (>0.80 = acceptable)
   - Demographic Parity Difference : 0.06 ✅
   - Equalized Odds : 0.08 ✅
   - Pas de pattern d'âge problématique

6. EXPLICABILITÉ
   - Top 3 features : years_experience (0.35), skill_match_count (0.25), nb_jobs (0.18)
   - SHAP : explications par candidat disponibles

7. LIMITES & RISQUES
   - Le modèle suggère, il ne décide pas
   - Données d'entraînement peuvent contenir biais historique
   - Monitoring recommandé tous les 3 mois
   - Retraining annuel conseillé

8. CONFORMITÉ
   - ✅ RGPD : données sensibles exclues
   - ✅ AI Act : décision finale humaine, explicabilité assurée
   - ✅ Rejet : toujours soumis à révision humaine avant notification
   - ✅ Contestation : le candidat peut demander révision

9. RECOMMANDATIONS
   - Utiliser le seuil 0.50
   - Marquer 0.40-0.60 comme "borderline, revue recommandée"
   - Auditer biais mensuellement
   - Afficher SHAP au recruteur pour chaque TOP 3 des scores
```

### 10.2 Traçabilité technique

```python
# model_metadata.json
{
  "version": "1.0",
  "trained_at": "2026-04-10T14:30:00Z",
  "trained_by": "MLOps Team",
  "data_snapshot": {
    "source": "data/processed/train.csv",
    "rows": 320,
    "features": ["years_experience", "nb_jobs", ...],
    "excluded_features": ["age", "gender", "name", "email"]
  },
  "model": {
    "type": "RandomForestClassifier",
    "params": {
      "n_estimators": 100,
      "max_depth": 10,
      "random_state": 42
    }
  },
  "performance": {
    "test_f1": 0.78,
    "test_precision": 0.80,
    "test_recall": 0.76,
    "test_roc_auc": 0.84
  },
  "fairness": {
    "disparate_impact_gender": 0.94,
    "demographic_parity_diff": 0.06,
    "equalized_odds_diff": 0.08
  },
  "decision_threshold": 0.50,
  "validation_date": "2026-04-05",
  "approved_by": ["hr_manager", "legal_team"]
}
```

### 10.3 Log de chaque prédiction

```python
# En production, logger :
log_entry = {
    'timestamp': datetime.now(),
    'cv_id': cv_id,
    'model_recommendation': recommendation,
    'model_probability': prob,
    'human_decision': 'HIRED' or 'REJECTED',  # Rempli par le recruteur
    'human_notes': '...',
    'disagreement': (recommendation != human_decision)
}

# Sauvegarder dans base de données
# Analyser mensuellement les désaccords
```

### 10.4 Checklist documentation

- ☐ Rapport ML final rédigé (1-2 pages exécutive, 5-10 pages technique)
- ☐ Config JSON complet (metadata, performance, fairness)
- ☐ Traçabilité du code : version Git du modèle
- ☐ Log des prédictions en production
- ☐ Monitorable : F1 recalculé mensuellement
- ☐ Approbation : RH, légal, Tech Lead

---

## 📋 Checklist condensée (TL;DR)

```
AVANT ENTRAÎNEMENT
☐ Data cleaned + no sensitive features
☐ Train/val/test split stratifié (60/20/20)
☐ Features scaled (StandardScaler)

ENTRAÎNEMENT
☐ Model choisi (LR MVP → RF → XGB)
☐ Hyperparamètres optimisés (grid search)
☐ Test F1 > 0.70

ÉVALUATION
☐ Precision, Recall, F1, ROC-AUC calculés
☐ Pas d'overfitting (learning curves)

BIAIS
☐ Disparate Impact Ratio > 0.80
☐ Demographic Parity & Equalized Odds < 0.10
☐ Pas d'anomalies par âge/genre

EXPLICABILITÉ
☐ Feature importance documentée
☐ SHAP testé sur exemples
☐ Top features = features métier (pas d'age/gender!)

VALIDATION
☐ Échantillon manuel approuve (>75%)
☐ Seuil décision documenté

DÉPLOIEMENT
☐ Modèle + scaler + config packagés
☐ API intégrée
☐ Monitoring setup

CONFORMITÉ
☐ Rapport ML rédigé
☐ Approuvé par RH + légal
☐ Log + traçabilité en place
```

---

## 🎯 Prochaines étapes

1. **Nettoyez et validez** les données pré-traitées (section 1)
2. **Splittez** train/val/test (section 2)
3. **Feature engineering** selon données réelles (section 3)
4. **Entraînez un MVP** (Régression Logistique) pour tester (section 4)
5. **Évaluez** sur test set (section 5)
6. **Auditez biais** sérieusement (section 6)
7. **Itérez** : Random Forest, ajustez features, revalidez
8. **Enfin** : XGBoost si performance nécessaire
9. **Déployez** avec monitoring (section 9)
10. **Documentez** tout (section 10)

Bonne chance ! 🚀

---

*Document créé avril 2026 — CV-Intelligence*
