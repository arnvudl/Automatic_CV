"""
Dashboard CV-Intelligence — Version Fairness-Aware
"""

import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, recall_score, precision_score, f1_score

# ==============================================================
# CONFIG
# ==============================================================
ROOT           = Path(__file__).parent.parent.parent
DATA_PATH      = ROOT / "data" / "processed" / "features.csv"
IDENTITIES_PATH= ROOT / "data" / "processed" / "identities.csv"
MODEL_PATH     = ROOT / "models" / "model.pkl"
SCALER_PATH    = ROOT / "models" / "scaler.pkl"
FEAT_PATH      = ROOT / "models" / "feature_cols.pkl"
THRESHOLD_PATH = ROOT / "models" / "threshold.pkl"
THRESHOLD_JR   = ROOT / "models" / "threshold_junior.pkl"

COUNTRY_PREFIXES = {
    '1': 'USA/Canada', '234': 'Nigeria', '31': 'Pays-Bas', '33': 'France',
    '351': 'Portugal', '353': 'Irlande', '39': 'Italie',
    '48': 'Pologne', '49': 'Allemagne', '91': 'Inde',
}

COLORS = {
    "primary": "#2563EB", "success": "#16A34A", "warning": "#D97706",
    "danger": "#DC2626", "light_bg": "#F1F5F9", "card_bg": "#FFFFFF",
    "text": "#1E293B", "muted": "#64748B",
}

# ==============================================================
# DATA LOADING
# ==============================================================
def get_country(phone):
    if not phone or not str(phone).startswith('+'): return "Autre"
    p = str(phone)[1:]
    for length in [3, 2, 1]:
        if p[:length] in COUNTRY_PREFIXES:
            return COUNTRY_PREFIXES[p[:length]]
    return "Autre"

def load_data():
    model     = joblib.load(MODEL_PATH)
    scaler    = joblib.load(SCALER_PATH)
    features  = joblib.load(FEAT_PATH)
    threshold = joblib.load(THRESHOLD_PATH) if THRESHOLD_PATH.exists() else 0.5
    thr_jr    = joblib.load(THRESHOLD_JR)   if THRESHOLD_JR.exists()   else threshold

    df      = pd.read_csv(DATA_PATH)
    df_id   = pd.read_csv(IDENTITIES_PATH)[['cv_id', 'gender', 'age', 'phone']]
    df      = df.merge(df_id, on='cv_id', how='left')

    target  = "label" if "label" in df.columns else "passed_next_stage"
    df      = df[df[target].notna()].copy()

    df['age_num']   = pd.to_numeric(df['age'], errors='coerce').fillna(30)
    df['is_junior'] = df['age_num'] < 30
    df['age_group'] = df['age_num'].apply(lambda x: "Jeune (<30)" if x < 30 else "Adulte (30+)")
    df['country']   = df['phone'].apply(get_country)

    X   = df[features].fillna(0).values.astype(float)
    X_s = scaler.transform(X)
    y_true  = df[target].astype(int).values
    y_proba = model.predict_proba(X_s)[:, 1]
    y_pred  = np.where(df['is_junior'].values, (y_proba >= thr_jr).astype(int), (y_proba >= threshold).astype(int))

    df['y_true']  = y_true
    df['y_proba'] = y_proba
    df['y_pred']  = y_pred

    return df, model, scaler, features, threshold, thr_jr, X_s, y_true, y_proba, y_pred

df, model, scaler, features, threshold, thr_jr, X_s, y_true, y_proba, y_pred = load_data()

# ==============================================================
# FIGURES
# ==============================================================

# --- KPIs ---
total_cv        = len(df)
invite_labels   = y_true.mean()
invite_model    = y_pred.mean()
fpr_, tpr_, _   = roc_curve(y_true, y_proba)
auc_score       = auc(fpr_, tpr_)
f1              = f1_score(y_true, y_pred)
cm              = confusion_matrix(y_true, y_pred)
recall_jr       = recall_score(df.loc[df['is_junior'], 'y_true'], df.loc[df['is_junior'], 'y_pred'], zero_division=0)
recall_ad       = recall_score(df.loc[~df['is_junior'], 'y_true'], df.loc[~df['is_junior'], 'y_pred'], zero_division=0)

# 1. Matrice de confusion
fig_cm = go.Figure(go.Heatmap(
    z=cm, x=['Prédit : Rejeté', 'Prédit : Invité'], y=['Réel : Rejeté', 'Réel : Invité'],
    text=[[str(v) for v in row] for row in cm], texttemplate="%{text}",
    colorscale='Blues', showscale=False,
))
fig_cm.update_layout(
    title="Matrice de Confusion", title_font_size=16,
    plot_bgcolor=COLORS['card_bg'], paper_bgcolor=COLORS['card_bg'],
    font_color=COLORS['text'], margin=dict(t=50, b=20, l=20, r=20),
    height=320,
)

# 2. Courbe ROC
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr_, y=tpr_, mode='lines', line=dict(color=COLORS['primary'], width=2.5),
                             name=f"AUC = {auc_score:.3f}"))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='gray', dash='dash'), name="Aléatoire"))
fig_roc.update_layout(
    title="Courbe ROC — Capacité de Tri", title_font_size=16,
    xaxis_title="Faux Positifs", yaxis_title="Vrais Positifs",
    legend=dict(x=0.6, y=0.1), height=320,
    plot_bgcolor=COLORS['card_bg'], paper_bgcolor=COLORS['card_bg'],
    font_color=COLORS['text'], margin=dict(t=50, b=40, l=50, r=20),
)

# 3. SHAP importance
import shap
explainer   = shap.LinearExplainer(model, X_s)
shap_vals   = explainer.shap_values(X_s)
mean_shap   = np.abs(shap_vals).mean(axis=0)
feat_df     = pd.DataFrame({'Variable': features, 'Impact SHAP': mean_shap}).sort_values('Impact SHAP')
FEAT_LABELS = {
    'years_experience': "Années d'expérience", 'education_level': "Niveau d'études",
    'has_multiple_languages': "Plurilingue (≥2 langues)", 'career_depth': "Profondeur de carrière",
    'potential_score': "Score Potentiel", 'is_it': "Secteur IT", 'avg_job_duration': "Durée moy. par poste",
    'junior_potential': "Potentiel Junior",
}
feat_df['Label'] = feat_df['Variable'].map(FEAT_LABELS).fillna(feat_df['Variable'])
fig_shap = px.bar(feat_df, x='Impact SHAP', y='Label', orientation='h',
                  color='Impact SHAP', color_continuous_scale='Blues',
                  title="Quelles variables influencent le modèle ?")
fig_shap.update_coloraxes(showscale=False)
fig_shap.update_layout(
    height=340, plot_bgcolor=COLORS['card_bg'], paper_bgcolor=COLORS['card_bg'],
    font_color=COLORS['text'], margin=dict(t=50, b=20, l=160, r=20), title_font_size=16,
    yaxis_title="", xaxis_title="Impact moyen (SHAP)",
)

# 4. Équité âge — avant / après
recall_jr_before = 0.26
fig_age = go.Figure()
fig_age.add_trace(go.Bar(name='Avant correction', x=["Adulte (30+)", "Jeune (<30)"],
                         y=[0.88, recall_jr_before], marker_color=['#93C5FD', '#FCA5A5']))
fig_age.add_trace(go.Bar(name='Après correction', x=["Adulte (30+)", "Jeune (<30)"],
                         y=[round(recall_ad, 2), round(recall_jr, 2)], marker_color=[COLORS['primary'], COLORS['success']]))
fig_age.update_layout(
    barmode='group', title="Équité par Âge — Recall (avant / après)", title_font_size=16,
    yaxis=dict(range=[0, 1.1], title="Recall"), xaxis_title="",
    height=320, plot_bgcolor=COLORS['card_bg'], paper_bgcolor=COLORS['card_bg'],
    font_color=COLORS['text'], margin=dict(t=50, b=30, l=50, r=20),
    legend=dict(x=0.65, y=0.95),
)

# 5. Équité genre
recall_gender = df.groupby('gender')[['y_true','y_pred']].apply(
    lambda g: pd.Series({'Recall': recall_score(g['y_true'], g['y_pred'], zero_division=0), 'n': len(g)})
).reset_index()
fig_gender = px.bar(recall_gender, x='gender', y='Recall', text='Recall',
                    color='gender', color_discrete_map={'Female': '#F9A8D4', 'Male': '#BAE6FD'},
                    title="Équité Homme / Femme (Recall)")
fig_gender.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_gender.update_layout(
    height=300, showlegend=False, yaxis=dict(range=[0, 1.2]),
    plot_bgcolor=COLORS['card_bg'], paper_bgcolor=COLORS['card_bg'],
    font_color=COLORS['text'], margin=dict(t=50, b=30, l=50, r=20),
    title_font_size=16, xaxis_title="", yaxis_title="Recall",
)

# 6. Équité pays
recall_country = df.groupby('country')[['y_true','y_pred']].apply(
    lambda g: pd.Series({'Recall': recall_score(g['y_true'], g['y_pred'], zero_division=0), 'n': len(g)})
).reset_index().sort_values('Recall', ascending=True)
fig_country = px.bar(recall_country, x='Recall', y='country', orientation='h',
                     text='Recall', color='Recall', color_continuous_scale='RdYlGn',
                     range_color=[0, 1], title="Équité par Pays (Recall)")
fig_country.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_country.update_coloraxes(showscale=False)
fig_country.update_layout(
    height=380, plot_bgcolor=COLORS['card_bg'], paper_bgcolor=COLORS['card_bg'],
    font_color=COLORS['text'], margin=dict(t=50, b=20, l=100, r=60),
    title_font_size=16, yaxis_title="", xaxis=dict(range=[0, 1.2]),
)

# 7. Distribution des scores par label
fig_dist = go.Figure()
for label, name, color in [(0, "Rejeté", "#FCA5A5"), (1, "Invité", "#86EFAC")]:
    subset = y_proba[y_true == label]
    fig_dist.add_trace(go.Histogram(x=subset, name=name, marker_color=color, opacity=0.75,
                                    xbins=dict(size=0.05)))
fig_dist.add_vline(x=threshold, line_dash="dash", line_color=COLORS['primary'],
                   annotation_text=f"Seuil adultes {threshold:.2f}", annotation_position="top right")
fig_dist.add_vline(x=thr_jr, line_dash="dot", line_color=COLORS['warning'],
                   annotation_text=f"Seuil juniors {thr_jr:.2f}", annotation_position="top left")
fig_dist.update_layout(
    barmode='overlay', title="Distribution des Scores de Probabilité", title_font_size=16,
    xaxis_title="Score (probabilité d'invitation)", yaxis_title="Nombre de CVs",
    height=320, plot_bgcolor=COLORS['card_bg'], paper_bgcolor=COLORS['card_bg'],
    font_color=COLORS['text'], margin=dict(t=50, b=40, l=50, r=20),
    legend=dict(x=0.02, y=0.95),
)

# ==============================================================
# LAYOUT HELPERS
# ==============================================================
def kpi_card(label, value, sub=None, color=COLORS['primary']):
    return html.Div([
        html.Div(label, style={'fontSize': '13px', 'color': COLORS['muted'], 'marginBottom': '4px', 'fontWeight': '500'}),
        html.Div(value, style={'fontSize': '32px', 'fontWeight': '700', 'color': color}),
        html.Div(sub or "", style={'fontSize': '12px', 'color': COLORS['muted'], 'marginTop': '4px'}),
    ], style={
        'background': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '20px 24px',
        'flex': '1', 'minWidth': '160px', 'boxShadow': '0 1px 6px rgba(0,0,0,0.08)',
        'borderTop': f'4px solid {color}',
    })

def section_title(text):
    return html.H2(text, style={
        'fontSize': '18px', 'fontWeight': '700', 'color': COLORS['text'],
        'marginTop': '36px', 'marginBottom': '12px', 'borderBottom': f'2px solid {COLORS["primary"]}',
        'paddingBottom': '6px',
    })

def graph_card(fig, width='48%'):
    return html.Div(dcc.Graph(figure=fig, config={'displayModeBar': False}), style={
        'background': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '8px',
        'boxShadow': '0 1px 6px rgba(0,0,0,0.08)', 'width': width,
    })

def info_box(text, color=COLORS['primary']):
    return html.Div(text, style={
        'background': f'{color}15', 'border': f'1px solid {color}40',
        'borderRadius': '8px', 'padding': '12px 16px',
        'fontSize': '14px', 'color': COLORS['text'], 'marginBottom': '12px',
    })

# ==============================================================
# APP LAYOUT
# ==============================================================
app = dash.Dash(__name__, title="CV-Intelligence Dashboard")
app.layout = html.Div(style={'background': COLORS['light_bg'], 'minHeight': '100vh', 'fontFamily': 'Inter, system-ui, sans-serif'}, children=[

    # Header
    html.Div([
        html.Div([
            html.H1("CV-Intelligence", style={'fontSize': '24px', 'fontWeight': '800', 'color': 'white', 'margin': '0'}),
            html.Div("Dashboard de performance — Tom & Arnaud", style={'fontSize': '13px', 'color': '#CBD5E1', 'marginTop': '2px'}),
        ]),
        html.Div(f"500 CVs · AUC {auc_score:.3f} · Fairness-Aware", style={
            'background': '#FFFFFF20', 'borderRadius': '20px', 'padding': '6px 16px',
            'fontSize': '13px', 'color': 'white', 'alignSelf': 'center',
        }),
    ], style={
        'background': f'linear-gradient(135deg, {COLORS["primary"]}, #1D4ED8)',
        'padding': '20px 32px', 'display': 'flex', 'justifyContent': 'space-between',
        'alignItems': 'center',
    }),

    html.Div(style={'padding': '24px 32px', 'maxWidth': '1400px', 'margin': '0 auto'}, children=[

        # KPIs
        section_title("Vue d'ensemble"),
        html.Div([
            kpi_card("CVs analysés", f"{total_cv}", "tous labellisés", COLORS['primary']),
            kpi_card("AUC-ROC", f"{auc_score:.3f}", "capacité de tri (max = 1)", COLORS['success']),
            kpi_card("F1-Score (Invité)", f"{f1:.2f}", "équilibre précision/recall", COLORS['warning']),
            kpi_card("Taux invitation labels", f"{invite_labels:.0%}", "référence terrain", COLORS['text']),
            kpi_card("Taux invitation modèle", f"{invite_model:.0%}", f"+{(invite_model - invite_labels):.0%} vs labels", COLORS['danger']),
        ], style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}),

        info_box(
            f"Le modèle invite {invite_model:.0%} des candidats contre {invite_labels:.0%} dans les labels réels. "
            f"L'écart (+{(invite_model - invite_labels):.0%}) vient de la correction d'équité : on abaisse le seuil pour les "
            f"moins de 30 ans (seuil junior = {thr_jr:.2f} vs {threshold:.2f} pour les adultes) afin de ne pas pénaliser "
            f"les jeunes à cause de leurs moins d'années d'expérience.",
            COLORS['primary']
        ),

        # Performance
        section_title("Performance du Modèle"),
        html.Div([graph_card(fig_cm), graph_card(fig_roc)],
                 style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}),

        html.Div(style={'marginTop': '16px'}, children=[graph_card(fig_dist, width='100%')]),

        info_box(
            "Lecture de la matrice : les cases diagonales sont les bonnes prédictions. "
            f"Sur {total_cv} CVs, le modèle trouve correctement {cm[1][1]} des {cm[1][0]+cm[1][1]} vrais invités (recall = {cm[1][1]/(cm[1][0]+cm[1][1]):.0%}), "
            f"mais génère {cm[0][1]} faux positifs (candidats invités par erreur).",
            COLORS['muted']
        ),

        # Variables
        section_title("Variables du Modèle"),
        html.Div([graph_card(fig_shap, width='100%')],
                 style={'display': 'flex', 'gap': '16px'}),
        info_box(
            "L'expérience et le niveau d'études dominent le modèle. "
            "'Score Potentiel' = (compétences + méthodes + certifs) / (expérience + 1) : "
            "permet de valoriser les profils qui progressent vite même avec peu d'ancienneté. "
            "'Potentiel Junior' booste spécifiquement les moins de 3 ans d'expérience à fort potentiel.",
            COLORS['primary']
        ),

        # Équité
        section_title("Audit d'Équité"),
        html.Div([graph_card(fig_age), graph_card(fig_gender)],
                 style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}),

        info_box(
            f"Sans correction, le recall des juniors était de 0.26 (le modèle ratait 74% des bons jeunes candidats). "
            f"Avec le seuil abaissé à {thr_jr:.2f}, il passe à {recall_jr:.2f}. "
            f"Les adultes passent de 0.88 à {recall_ad:.2f} — légère baisse car davantage de juniors sont invités et "
            "occupent une part du quota.",
            COLORS['success']
        ),

        html.Div(style={'marginTop': '16px'}, children=[graph_card(fig_country, width='100%')]),
        info_box(
            "Le modèle ne connaît pas le pays du candidat — les écarts de recall par pays reflètent des "
            "différences de profils (expérience, diplôme, secteur) selon les origines géographiques dans le dataset, "
            "pas un biais explicite sur la nationalité.",
            COLORS['muted']
        ),

        # Footer
        html.Div("CV-Intelligence — Tom Perez Le Tiec & Arnaud Leroy · Avril 2026", style={
            'textAlign': 'center', 'color': COLORS['muted'], 'fontSize': '12px',
            'marginTop': '40px', 'paddingTop': '16px', 'borderTop': f'1px solid #E2E8F0',
        }),
    ])
])

if __name__ == '__main__':
    app.run(debug=True, port=8050)
