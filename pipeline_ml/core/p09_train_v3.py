"""
p09_train_v3.py — Entraînement v3 (Anti-Bias)
Features v3 : years_experience remplacé par exp_per_year_of_age + ajout field_match.
Objectif : réduire le biais structurel genre sans toucher aux seuils de décision.

Compare les métriques d'équité v5 (baseline) vs v3 sur le même split test.
Sauvegarde le modèle v3 dans models/ si amélioration confirmée.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score,
    precision_recall_curve, classification_report
)

ROOT           = Path(__file__).parent.parent.parent
FEATURES_PATH  = ROOT / "data" / "processed" / "features.csv"
IDENTITIES_PATH= ROOT / "data" / "processed" / "identities.csv"
MODELS_DIR     = ROOT / "models"
REPORTS_DIR    = ROOT / "reports"
RANDOM_STATE   = 42

# ── Baseline v5 ──────────────────────────────────────────────────
V5_FEATURES = [
    "years_experience", "avg_job_duration", "education_level",
    "potential_score", "junior_potential", "has_multiple_languages",
    "career_depth", "is_it",
]

# ── Nouvelles features v3 ────────────────────────────────────────
# years_experience → exp_per_year_of_age (normalisé par durée carrière possible)
# + field_match (pertinence formation / secteur)
V3_FEATURES = [
    "exp_per_year_of_age", "avg_job_duration", "education_level",
    "potential_score", "junior_potential", "has_multiple_languages",
    "career_depth", "is_it", "field_match",
]

TARGET_COL = "label"


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    df_id = pd.read_csv(IDENTITIES_PATH)[["cv_id", "gender", "age"]]
    df_id["age"] = pd.to_numeric(df_id["age"], errors="coerce").fillna(30)
    df = df.merge(df_id, on="cv_id", how="left")
    df = df[df[TARGET_COL].notna()].copy()
    df["is_junior"] = (df["age"] < 30).astype(int)
    df["gender"] = df["gender"].fillna("Unknown")
    return df


def best_threshold_f1(y_true, y_proba):
    p, r, t = precision_recall_curve(y_true, y_proba)
    f1 = 2 * p * r / (p + r + 1e-9)
    return float(t[np.argmax(f1[:-1])])


def best_threshold_recall(y_true, y_proba, recall_target=0.55):
    p, r, t = precision_recall_curve(y_true, y_proba)
    valid = r[:-1] >= recall_target
    if valid.any():
        return float(t[valid][np.argmax(p[:-1][valid])])
    return best_threshold_f1(y_true, y_proba)


def train_and_eval(X_train, X_test, y_train, y_test,
                   jr_train, jr_test, gender_test, feature_names, label):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE,
                                class_weight="balanced")
    model.fit(X_tr, y_train)

    y_proba_tr = model.predict_proba(X_tr)[:, 1]
    y_proba_te = model.predict_proba(X_te)[:, 1]

    # Seuils différenciés âge (même logique que v5)
    adult_mask = jr_train == 0
    thr_adult  = best_threshold_f1(y_train[adult_mask], y_proba_tr[adult_mask])
    jr_mask    = jr_train == 1
    thr_junior = best_threshold_recall(y_train[jr_mask], y_proba_tr[jr_mask])

    y_pred = np.where(jr_test == 1,
                      (y_proba_te >= thr_junior).astype(int),
                      (y_proba_te >= thr_adult).astype(int))

    auc = roc_auc_score(y_test, y_proba_te)

    lines = [f"── {label} ─────────────────────────────────────────────────────"]
    lines.append(f"  Features ({len(feature_names)}) : {feature_names}")
    lines.append(f"  Seuil adultes : {thr_adult:.3f}  |  Seuil juniors : {thr_junior:.3f}")
    lines.append(f"  AUC : {auc:.3f}")
    lines.append("  Par genre :")
    for g in ["Female", "Male"]:
        m = gender_test == g
        rec  = recall_score(y_test[m], y_pred[m], zero_division=0)
        prec = precision_score(y_test[m], y_pred[m], zero_division=0)
        lines.append(f"    {g:8}  n={m.sum():3}  Recall={rec:.3f}  Precision={prec:.3f}")
    lines.append("  Par âge :")
    for is_jr, lbl in [(1, "Junior (<30)"), (0, "Adulte (30+)")]:
        m = jr_test == is_jr
        rec  = recall_score(y_test[m], y_pred[m], zero_division=0)
        prec = precision_score(y_test[m], y_pred[m], zero_division=0)
        lines.append(f"    {lbl:14}  n={m.sum():3}  Recall={rec:.3f}  Precision={prec:.3f}")

    # SHAP-like : coefficients × std (importance approximée)
    importances = np.abs(model.coef_[0]) * X_tr.std(axis=0)
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    lines.append("  Importance features (|coef| × std) :")
    for feat, val in imp_series.items():
        lines.append(f"    {feat:28} {val:.4f}")

    return model, scaler, y_pred, y_proba_te, thr_adult, thr_junior, lines


def main():
    df = load_data()
    y   = df[TARGET_COL].astype(int).values
    jr  = df["is_junior"].values
    gen = df["gender"].values

    # Même split pour comparaison équitable
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2,
                                       random_state=RANDOM_STATE, stratify=y)

    y_tr, y_te   = y[tr_idx],  y[te_idx]
    jr_tr, jr_te = jr[tr_idx], jr[te_idx]
    gen_te       = gen[te_idx]

    report_lines = [
        "RAPPORT ENTRAÎNEMENT V3 — Features Anti-Biais",
        "=" * 60,
        f"Dataset : {len(df)} CV  |  Train : {len(tr_idx)}  |  Test : {len(te_idx)}",
        f"Positifs test : {y_te.sum()} ({y_te.mean():.1%})",
        "",
    ]

    # ── Baseline v5 ──────────────────────────────────────────────
    X_v5 = df[V5_FEATURES].fillna(0).values
    _, _, pred_v5, proba_v5, _, _, lines_v5 = train_and_eval(
        X_v5[tr_idx], X_v5[te_idx], y_tr, y_te, jr_tr, jr_te, gen_te,
        V5_FEATURES, "BASELINE v5"
    )
    report_lines += lines_v5 + [""]

    # ── Modèle v3 ─────────────────────────────────────────────────
    X_v3 = df[V3_FEATURES].fillna(0).values
    model_v3, scaler_v3, pred_v3, proba_v3, thr_a, thr_j, lines_v3 = train_and_eval(
        X_v3[tr_idx], X_v3[te_idx], y_tr, y_te, jr_tr, jr_te, gen_te,
        V3_FEATURES, "MODÈLE v3 (exp_per_year_of_age + field_match)"
    )
    report_lines += lines_v3 + [""]

    # ── Résumé comparatif ─────────────────────────────────────────
    report_lines += [
        "── RÉSUMÉ COMPARATIF ─────────────────────────────────────────",
        f"{'Métrique':35} {'v5 (baseline)':>15} {'v3 (anti-biais)':>16} {'Δ':>8}",
    ]
    metrics = [
        ("AUC global",
         roc_auc_score(y_te, proba_v5),
         roc_auc_score(y_te, proba_v3)),
        ("Recall Female",
         recall_score(y_te[gen_te == "Female"], pred_v5[gen_te == "Female"], zero_division=0),
         recall_score(y_te[gen_te == "Female"], pred_v3[gen_te == "Female"], zero_division=0)),
        ("Recall Male",
         recall_score(y_te[gen_te == "Male"], pred_v5[gen_te == "Male"], zero_division=0),
         recall_score(y_te[gen_te == "Male"], pred_v3[gen_te == "Male"], zero_division=0)),
        ("Écart genre (Male - Female)",
         recall_score(y_te[gen_te == "Male"], pred_v5[gen_te == "Male"], zero_division=0) -
         recall_score(y_te[gen_te == "Female"], pred_v5[gen_te == "Female"], zero_division=0),
         recall_score(y_te[gen_te == "Male"], pred_v3[gen_te == "Male"], zero_division=0) -
         recall_score(y_te[gen_te == "Female"], pred_v3[gen_te == "Female"], zero_division=0)),
        ("Recall Junior",
         recall_score(y_te[jr_te == 1], pred_v5[jr_te == 1], zero_division=0),
         recall_score(y_te[jr_te == 1], pred_v3[jr_te == 1], zero_division=0)),
        ("Precision Junior",
         precision_score(y_te[jr_te == 1], pred_v5[jr_te == 1], zero_division=0),
         precision_score(y_te[jr_te == 1], pred_v3[jr_te == 1], zero_division=0)),
    ]
    for name, v5_val, v3_val in metrics:
        delta = v3_val - v5_val
        sign  = "+" if delta >= 0 else ""
        report_lines.append(f"  {name:33} {v5_val:>15.3f} {v3_val:>16.3f} {sign}{delta:>7.3f}")

    # Écriture rapport
    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "train_v3.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Rapport ecrit : {report_path}")

    # Sauvegarde modèle v3
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model_v3,  MODELS_DIR / "model_v3.pkl")
    joblib.dump(scaler_v3, MODELS_DIR / "scaler_v3.pkl")
    joblib.dump(V3_FEATURES, MODELS_DIR / "feature_cols_v3.pkl")
    joblib.dump(thr_a,     MODELS_DIR / "threshold_v3.pkl")
    joblib.dump(thr_j,     MODELS_DIR / "threshold_junior_v3.pkl")
    print(f"\nModèle v3 sauvegardé dans {MODELS_DIR}")


if __name__ == "__main__":
    main()
