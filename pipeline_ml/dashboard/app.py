import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, precision_recall_curve
)
from sklearn.utils import resample

# ==============================================================
# CONFIG
# ==============================================================
ROOT = Path(__file__).parent.parent.parent
DATA_PATH      = ROOT / "data" / "processed" / "features.csv"
IDENTITIES_PATH = ROOT / "data" / "processed" / "identities.csv"
STUDENT_LABELS = ROOT / "pipeline_ml" / "config" / "student_labels.csv"
MODEL_PATH     = ROOT / "models" / "model.pkl"
SCALER_PATH    = ROOT / "models" / "scaler.pkl"
FEAT_PATH      = ROOT / "models" / "feature_cols.pkl"
THRESHOLD_PATH = ROOT / "models" / "threshold.pkl"

def load_assets():
    if not MODEL_PATH.exists() or not DATA_PATH.exists():
        return None
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEAT_PATH)
    threshold = joblib.load(THRESHOLD_PATH) if THRESHOLD_PATH.exists() else 0.55
    
    df = pd.read_csv(DATA_PATH)
    df_id = pd.read_csv(IDENTITIES_PATH)[['cv_id', 'source_filename']]
    df = df.merge(df_id, on='cv_id', how='left')
    
    # Comparaison avec student_labels.csv
    if STUDENT_LABELS.exists():
        df_sl = pd.read_csv(STUDENT_LABELS)
        df = df.merge(df_sl, left_on='source_filename', right_on='filename', how='left', suffixes=('', '_original'))

    target = "label" if "label" in df.columns else "passed_next_stage"
    df = df[df[target].notna()].copy()
    
    return model, scaler, features, df, target, threshold

def get_dashboard_data():
    assets = load_assets()
    if not assets: return None
    
    model, scaler, features, df, target, threshold = assets
    X = df[features].values
    y_true = df[target].values.astype(int)
    
    X_s = scaler.transform(X)
    y_proba = model.predict_proba(X_s)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # 1. Matrice de Confusion (Mise à jour)
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, x=['Rejeté', 'Invité'], y=['Rejeté', 'Invité'],
                       title=f"Matrice de Confusion (v5 - Seuil: {threshold:.3f})",
                       color_continuous_scale='RdBu_r')

    # 2. Importance des Features
    importances = model.coef_[0]
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
    fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Importance des Variables")

    # 3. Comparaison avec Student Labels Original
    fig_comp = None
    if 'passed_next_stage_original' in df.columns:
        y_orig = df['passed_next_stage_original'].fillna(0).astype(int)
        comp_df = pd.DataFrame({
            'Source': ['Modèle IA', 'Labels Étudiants'],
            'Taux Invitation': [y_pred.mean(), y_orig.mean()]
        })
        fig_comp = px.bar(comp_df, x='Source', y='Taux Invitation', color='Source', 
                          title="Comparaison : Sévérité IA vs Labels Étudiants")

    return {
        'cm': fig_cm, 'feat': fig_feat, 'comp': fig_comp, 'stats': f"F1-Score: {f1_score(y_true, y_pred):.3f} | Accuracy: {accuracy_score(y_true, y_pred):.3f}"
    }

# (Le reste du code Dash est identique mais simplifié pour la clarté)
app = dash.Dash(__name__, title="CV-Intelligence Dashboard v5")
app.layout = html.Div([
    html.H1("🚀 Dashboard de Performance v5 (Tom & Arnaud)"),
    html.Div(id='stats-summary', children="Chargement..."),
    html.Div([
        dcc.Graph(id='cm-graph'),
        dcc.Graph(id='feat-graph'),
        dcc.Graph(id='comp-graph'),
    ], style={'display': 'flex', 'flexWrap': 'wrap'})
])

@app.callback(
    [dash.Output('cm-graph', 'figure'),
     dash.Output('feat-graph', 'figure'),
     dash.Output('comp-graph', 'figure'),
     dash.Output('stats-summary', 'children')],
    [dash.Input('cm-graph', 'id')] # Trigger simple
)
def update_graphs(_):
    data = get_dashboard_data()
    return data['cm'], data['feat'], data['comp'], data['stats']

if __name__ == '__main__':
    app.run(debug=True, port=8050)
