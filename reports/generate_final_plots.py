import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, recall_score

# CONFIG
ROOT = Path("..")
FEATURES_PATH   = ROOT / "data" / "processed" / "features.csv"
IDENTITIES_PATH = ROOT / "data" / "processed" / "identities.csv"
REPORTS_DIR     = ROOT / "reports" / "plots"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# CHARGEMENT ASSETS
df_feat = pd.read_csv(FEATURES_PATH)
df_id = pd.read_csv(IDENTITIES_PATH)[['cv_id', 'gender', 'age', 'phone']]
df = df_feat.merge(df_id, on='cv_id', how='left')

COUNTRY_PREFIXES = {
    '1': 'USA/Canada',
    '234': 'Nigeria',
    '31': 'Pays-Bas',
    '33': 'France',
    '351': 'Portugal',
    '353': 'Irlande',
    '39': 'Italie',
    '48': 'Pologne',
    '49': 'Allemagne',
    '91': 'Inde',
}

def get_country(phone):
    if not phone or not str(phone).startswith('+'): return "Inconnu"
    p = str(phone)[1:]
    for length in [3, 2, 1]:
        prefix = p[:length]
        if prefix in COUNTRY_PREFIXES: return COUNTRY_PREFIXES[prefix]
    return "Autre"

df["country"] = df["phone"].apply(get_country)

model = joblib.load(ROOT / "models" / "model.pkl")
scaler = joblib.load(ROOT / "models" / "scaler.pkl")
features = joblib.load(ROOT / "models" / "feature_cols.pkl")
threshold = joblib.load(ROOT / "models" / "threshold.pkl")

target = "label" if "label" in df.columns else "passed_next_stage"
df = df[df[target].notna()].copy()
X = df[features].values
X_s = scaler.transform(X)
y_true = df[target].values.astype(int)
y_proba = model.predict_proba(X_s)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

# 1. MATRICE DE CONFUSION
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Rejeté', 'Invité'], yticklabels=['Rejeté', 'Invité'])
plt.title(f"Matrice de Confusion (v6 - Dataset Réel)")
plt.ylabel('Réel')
plt.xlabel('Prédit')
plt.savefig(REPORTS_DIR / "confusion_matrix.png")
plt.close()

# 2. COURBE ROC
fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC area = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("Capacité de Tri (AUC-ROC)")
plt.xlabel("Taux de Faux Positifs")
plt.ylabel("Taux de Vrais Positifs")
plt.legend(loc="lower right")
plt.savefig(REPORTS_DIR / "roc_curve.png")
plt.close()

# 3. ÉQUITÉ (RECALL PAR GROUPE)
df['age_group'] = df['age'].apply(lambda x: "Jeune (<30)" if x < 30 else "Adulte (30+)")
def get_recall(x):
    yt = x[target].values.astype(int)
    if len(yt) < 1: return 0.0
    yp = (model.predict_proba(scaler.transform(x[features].values))[:,1] >= threshold).astype(int)
    return recall_score(yt, yp, zero_division=0)

recall_gender = df.groupby('gender').apply(get_recall)
recall_age = df.groupby('age_group').apply(get_recall)
recall_country = df.groupby('country').apply(get_recall)

# Figure 1: Genre & Age
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
recall_gender.plot(kind='bar', ax=ax1, color=['pink', 'lightblue'])
ax1.set_title("Équité Homme/Femme (Recall)")
ax1.set_ylim(0, 1.1)
recall_age.plot(kind='bar', ax=ax2, color=['orange', 'green'])
ax2.set_title("Équité Jeune/Adulte (Recall)")
ax2.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "fairness_metrics.png")
plt.close()

# Figure 2: Pays
plt.figure(figsize=(12, 6))
recall_country.sort_values(ascending=False).plot(kind='bar', color='teal')
plt.title("Équité par Pays (Recall)")
plt.ylabel("Recall")
plt.ylim(0, 1.1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "fairness_country.png")
plt.close()

# 4. FEATURE IMPORTANCE (SHAP)
explainer = shap.LinearExplainer(model, X_s)
shap_values = explainer.shap_values(X_s)
mean_shap = np.abs(shap_values).mean(axis=0)
feat_imp = pd.Series(mean_shap, index=features).sort_values()
plt.figure(figsize=(8, 5))
feat_imp.plot(kind='barh', color='steelblue')
plt.title("Importance des Variables (SHAP moyen)")
plt.xlabel("Impact moyen sur la prédiction")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "feature_importance.png")
plt.close()

print("Graphiques finaux générés sur FEATURES.CSV + IDENTITIES.CSV")
