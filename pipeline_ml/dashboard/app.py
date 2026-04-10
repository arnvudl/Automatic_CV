import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample

# ==============================================================
# CONFIG (Adaptée pour dashboard/app.py)
# ==============================================================
ROOT = Path(__file__).parent.parent.parent
DATA_PATH      = ROOT / "data" / "processed" / "dataset.csv"
MODEL_PATH     = ROOT / "models" / "model.pkl"
SCALER_PATH    = ROOT / "models" / "scaler.pkl"
FEAT_PATH      = ROOT / "models" / "feature_cols.pkl"
THRESHOLD_PATH = ROOT / "models" / "threshold.pkl"

def load_all_assets():
    if not MODEL_PATH.exists() or not DATA_PATH.exists():
        return None
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEAT_PATH)
    df = pd.read_csv(DATA_PATH)
    
    threshold = 0.5
    if THRESHOLD_PATH.exists():
        threshold = joblib.load(THRESHOLD_PATH)
        
    target = "passed_next_stage" if "passed_next_stage" in df.columns else "label"
    df = df[df[target].notna()].copy()
    
    return model, scaler, features, df, target, threshold

def get_figures():
    assets = load_all_assets()
    if not assets: return None, None, None, "Erreur de chargement des assets."
    
    model, scaler, features, df, target, threshold = assets
    X = df[features].values
    y_true = df[target].values
    
    try:
        X_s = scaler.transform(X)
        y_proba = model.predict_proba(X_s)[:, 1]
    except:
        y_proba = model.predict_proba(X)[:, 1]
        
    y_pred = (y_proba >= threshold).astype(int)
    
    # CM
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, x=['Rejeté', 'Invité'], y=['Rejeté', 'Invité'],
                       title="Matrice de Confusion (Seuil: {:.3f})".format(threshold),
                       color_continuous_scale='Blues')
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name='ROC', mode='lines'))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash'), name='Baseline'))
    fig_roc.update_layout(title="Courbe ROC (AUC: {:.3f})".format(auc(fpr, tpr)))
    
    # Metrics Text
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    metrics_text = [
        html.P(f"Accuracy: {acc:.3f}"),
        html.P(f"F1-Score: {f1:.3f}"),
        html.P(f"Seuil appliqué: {threshold:.3f}")
    ]
    
    return fig_cm, fig_roc, metrics_text, None

app = dash.Dash(__name__)

def serve_layout():
    fig_cm, fig_roc, metrics_text, err = get_figures()
    if err: return html.Div([html.H1(err)])
    
    return html.Div([
        html.H1("Dashboard ML Dynamique — CV-Intelligence", style={'textAlign': 'center'}),
        html.Div([
            html.Div([dcc.Graph(figure=fig_cm)], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig_roc)], style={'width': '48%', 'display': 'inline-block'})
        ]),
        html.Div(metrics_text, style={'padding': '20px', 'fontSize': '20px'})
    ])

app.layout = serve_layout

if __name__ == '__main__':
    app.run(debug=True, port=8050)
