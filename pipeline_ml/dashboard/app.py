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
DATA_PATH      = ROOT / "data" / "processed" / "dataset.csv"
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
    df = pd.read_csv(DATA_PATH)
    
    threshold = 0.5
    if THRESHOLD_PATH.exists():
        threshold = joblib.load(THRESHOLD_PATH)
        
    target = "passed_next_stage" if "passed_next_stage" in df.columns else "label"
    df = df[df[target].notna()].copy()
    
    return model, scaler, features, df, target, threshold

def calculate_bootstrap(y_true, y_proba, threshold, n_iter=100):
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'AUC': []}
    n_size = len(y_true)
    
    for i in range(n_iter):
        idx = resample(np.arange(n_size), replace=True, n_samples=n_size, random_state=i)
        yt, yp_proba = y_true[idx], y_proba[idx]
        if len(np.unique(yt)) < 2: continue
        
        yp_pred = (yp_proba >= threshold).astype(int)
        metrics['Accuracy'].append(accuracy_score(yt, yp_pred))
        metrics['Precision'].append(precision_score(yt, yp_pred, zero_division=0))
        metrics['Recall'].append(recall_score(yt, yp_pred, zero_division=0))
        metrics['F1'].append(f1_score(yt, yp_pred, zero_division=0))
        fpr, tpr, _ = roc_curve(yt, yp_proba)
        metrics['AUC'].append(auc(fpr, tpr))
        
    return pd.DataFrame(metrics)

def get_dashboard_data():
    assets = load_assets()
    if not assets: return None
    
    model, scaler, features, df, target, threshold = assets
    X = df[features].values
    y_true = df[target].values
    
    try:
        X_s = scaler.transform(X)
        y_proba = model.predict_proba(X_s)[:, 1]
    except:
        y_proba = model.predict_proba(X)[:, 1]
    
    y_pred = (y_proba >= threshold).astype(int)
    
    # 1. Matrice de Confusion
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, x=['Rejeté', 'Invité'], y=['Rejeté', 'Invité'],
                       title=f"Matrice de Confusion (Seuil: {threshold:.3f})",
                       color_continuous_scale='RdBu_r')

    # 2. ROC & PR Curves
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'AUC: {auc(fpr, tpr):.3f}', fill='tozeroy'))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash'), name='Baseline'))
    fig_roc.update_layout(title="Courbe ROC", xaxis_title="FPR", yaxis_title="TPR")

    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    fig_pr = px.area(x=rec, y=prec, title="Courbe Précision-Rappel",
                     labels=dict(x="Rappel", y="Précision"))

    # 3. Importance des Features
    if hasattr(model, 'coef_'):
        importances = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(features))
    
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
    fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Importance des Variables")

    # 4. Distribution des Scores
    fig_dist = px.histogram(x=y_proba, color=y_true.astype(str), nbins=50, barmode='overlay',
                            title="Distribution des Probabilités Prédites",
                            labels={'x': 'Score (Probabilité)', 'color': 'Réel (0=Rejet, 1=Invité)'})
    fig_dist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Seuil")

    # 5. Bootstrap metrics
    boot_df = calculate_bootstrap(y_true, y_proba, threshold)
    fig_boot = px.violin(boot_df.melt(), y="value", x="variable", color="variable", box=True, points="all",
                         title="Stabilité des Métriques (Bootstrap + IC 95%)")
    
    stats = []
    for col in boot_df.columns:
        mean = boot_df[col].mean()
        low, high = np.percentile(boot_df[col], [2.5, 97.5])
        stats.append(html.Div([
            html.B(f"{col}: "), html.Span(f"{mean:.3f} "),
            html.Small(f"[{low:.3f} - {high:.3f}]", style={'color': 'gray'})
        ], style={'marginBottom': '5px'}))

    return {
        'cm': fig_cm, 'roc': fig_roc, 'pr': fig_pr, 'feat': fig_feat, 
        'dist': fig_dist, 'boot': fig_boot, 'stats': stats
    }

app = dash.Dash(__name__, title="CV-Intelligence Super-Dashboard")

def serve_layout():
    data = get_dashboard_data()
    if not data:
        return html.Div([html.H1("Erreur : Assets (modèle/data) manquants. Lancez le pipeline d'abord.")])

    return html.Div([
        html.Header([
            html.H1("🚀 Performance ML — CV-Intelligence", style={'margin': '0'}),
            html.P("Analyse approfondie, Bootstrap et stabilité du modèle", style={'margin': '5px 0 20px 0', 'opacity': '0.8'})
        ], style={'backgroundColor': '#2c3e50', 'color': 'white', 'padding': '20px', 'textAlign': 'center'}),

        html.Div([
            # Colonne Gauche : Stats & Importance
            html.Div([
                html.Div([
                    html.H3("🎯 Métriques (IC 95%)"),
                    html.Div(data['stats'])
                ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '10px', 'marginBottom': '20px', 'border': '1px solid #dee2e6'}),
                dcc.Graph(figure=data['feat'], style={'height': '500px'})
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

            # Colonne Droite : Graphiques
            html.Div([
                html.Div([
                    html.Div([dcc.Graph(figure=data['cm'])], style={'width': '50%', 'display': 'inline-block'}),
                    html.Div([dcc.Graph(figure=data['dist'])], style={'width': '50%', 'display': 'inline-block'}),
                ]),
                html.Div([
                    html.Div([dcc.Graph(figure=data['roc'])], style={'width': '50%', 'display': 'inline-block'}),
                    html.Div([dcc.Graph(figure=data['pr'])], style={'width': '50%', 'display': 'inline-block'}),
                ]),
                html.Div([dcc.Graph(figure=data['boot'])])
            ], style={'width': '68%', 'display': 'inline-block', 'padding': '10px'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'maxWidth': '1400px', 'margin': '0 auto'})
    ], style={'fontFamily': 'Segoe UI, Arial, sans-serif', 'backgroundColor': '#fff'})

app.layout = serve_layout

if __name__ == '__main__':
    app.run(debug=True, port=8050)
