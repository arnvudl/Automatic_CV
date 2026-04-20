"""
p08_fairness_v2.py — Corrections d'équité v2
  Étape 1 : Correction genre par reweighting (entraînement)
  Étape 2 : Correction genre par seuils différenciés (post-processing)
  Étape 3 : Amélioration précision juniors (ajustement objectif seuil)

Rapport : reports/fairness_v2.txt
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, precision_recall_curve, classification_report
)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent.parent
FEATURES_PATH  = ROOT / "data" / "processed" / "features.csv"
IDENTITIES_PATH= ROOT / "data" / "processed" / "identities.csv"
MODELS_DIR     = ROOT / "models"
REPORTS_DIR    = ROOT / "reports"
RANDOM_STATE   = 42

V5_FEATURES = [
    "years_experience", "avg_job_duration", "education_level",
    "potential_score", "junior_potential", "has_multiple_languages",
    "career_depth", "is_it"
]
TARGET_COL = "label"

# ──────────────────────────────────────────────
# CHARGEMENT
# ──────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEATURES_PATH)
    df_id = pd.read_csv(IDENTITIES_PATH)[["cv_id", "gender", "age"]]
    df = df.merge(df_id, on="cv_id", how="left")
    df = df[df[TARGET_COL].notna()].copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(30)
    df["is_junior"] = (df["age"] < 30).astype(int)
    return df

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def dual_threshold_predict(y_proba, is_junior, thr_adult, thr_junior):
    return np.where(is_junior, (y_proba >= thr_junior).astype(int),
                               (y_proba >= thr_adult).astype(int))

def best_threshold_recall_target(y_true, y_proba, recall_target=0.55):
    """Seuil qui atteint recall_target avec la meilleure précision possible."""
    p, r, t = precision_recall_curve(y_true, y_proba)
    valid = r[:-1] >= recall_target
    if valid.any():
        return float(t[valid][np.argmax(p[:-1][valid])])
    # Fallback : maximise F1
    f1 = 2 * p * r / (p + r + 1e-9)
    return float(t[np.argmax(f1[:-1])])

def best_threshold_f1(y_true, y_proba):
    p, r, t = precision_recall_curve(y_true, y_proba)
    f1 = 2 * p * r / (p + r + 1e-9)
    return float(t[np.argmax(f1[:-1])])

def group_metrics(y_true, y_pred, label):
    rec  = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return f"  {label:22} n={len(y_true):3}  Recall={rec:.3f}  Precision={prec:.3f}  F1={f1:.3f}"

# ──────────────────────────────────────────────
# BASELINE (modèle v5 actuel)
# ──────────────────────────────────────────────
def run_baseline(X_train_s, X_test_s, y_train, y_test, jr_test, df_test):
    model = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE,
                               class_weight="balanced")
    model.fit(X_train_s, y_train)

    y_proba = model.predict_proba(X_test_s)[:, 1]

    # Seuil adultes optimisé sur train
    adult_mask = (df_test["is_junior"] == 0)  # approximation sur test uniquement pour affichage
    thr_adult   = 0.614   # valeurs v5 connues
    thr_junior  = 0.374

    y_pred = dual_threshold_predict(y_proba, jr_test, thr_adult, thr_junior)
    return model, y_proba, y_pred, thr_adult, thr_junior

# ──────────────────────────────────────────────
# ÉTAPE 1 — Reweighting genre
# ──────────────────────────────────────────────
def run_step1_reweighting(X_train_s, X_test_s, y_train, y_test,
                           jr_train, jr_test, gender_train):
    """
    Upweighter les femmes pendant l'entraînement.
    Poids = recall_ratio × size_ratio pour compenser le déséquilibre observé.
    """
    # recall_ratio = recall_male / recall_female = 0.70 / 0.57 ≈ 1.23
    # size_ratio   = n_male / n_female ≈ 267/233 ≈ 1.15
    female_weight = 1.23  # recall gap compensation
    sample_weights = np.where(gender_train == "Female", female_weight, 1.0)

    model_rw = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE,
                                   class_weight="balanced")
    model_rw.fit(X_train_s, y_train, sample_weight=sample_weights)

    y_proba_rw = model_rw.predict_proba(X_test_s)[:, 1]

    # Recalibrer les seuils sur train
    y_proba_train = model_rw.predict_proba(X_train_s)[:, 1]
    adult_mask_train = jr_train == 0
    thr_adult = best_threshold_f1(y_train[adult_mask_train],
                                   y_proba_train[adult_mask_train])
    junior_mask_train = jr_train == 1
    thr_junior = best_threshold_recall_target(y_train[junior_mask_train],
                                               y_proba_train[junior_mask_train],
                                               recall_target=0.55)

    y_pred_rw = dual_threshold_predict(y_proba_rw, jr_test, thr_adult, thr_junior)
    return model_rw, y_proba_rw, y_pred_rw, thr_adult, thr_junior

# ──────────────────────────────────────────────
# ÉTAPE 2 — Seuils différenciés par genre (equalized recall)
# ──────────────────────────────────────────────
def run_step2_gender_thresholds(X_train_s, X_test_s, y_train, y_test,
                                 jr_train, jr_test, gender_train, gender_test,
                                 model_rw):
    """
    À partir du modèle reweighté (étape 1), on ajoute un seuil par genre
    pour égaliser davantage le recall. On combine : genre × âge → 4 seuils.
    """
    y_proba_train = model_rw.predict_proba(X_train_s)[:, 1]
    y_proba_test  = model_rw.predict_proba(X_test_s)[:, 1]

    thresholds = {}
    for gender in ["Male", "Female"]:
        for is_jr, jr_label in [(0, "adult"), (1, "junior")]:
            mask = (gender_train == gender) & (jr_train == is_jr)
            key = (gender, jr_label)
            if mask.sum() < 5 or y_train[mask].sum() < 2:
                thresholds[key] = 0.5
                continue
            if jr_label == "junior":
                thresholds[key] = best_threshold_recall_target(
                    y_train[mask], y_proba_train[mask], recall_target=0.55)
            else:
                thresholds[key] = best_threshold_f1(y_train[mask], y_proba_train[mask])

    def predict_4way(y_proba, is_junior, gender_arr):
        preds = np.zeros(len(y_proba), dtype=int)
        for g in ["Male", "Female"]:
            for jr_val, jr_label in [(0, "adult"), (1, "junior")]:
                mask = (gender_arr == g) & (is_junior == jr_val)
                if mask.any():
                    preds[mask] = (y_proba[mask] >= thresholds[(g, jr_label)]).astype(int)
        return preds

    y_pred_4way = predict_4way(y_proba_test, jr_test, gender_test)
    return y_pred_4way, thresholds

# ──────────────────────────────────────────────
# ÉTAPE 3 — Précision juniors
# ──────────────────────────────────────────────
def run_step3_junior_precision(X_train_s, X_test_s, y_train, y_test,
                                jr_train, jr_test, model_rw):
    """
    Remplace l'objectif recall≥0.60 par recall≥0.55 pour le seuil junior,
    ce qui récupère de la précision sans trop sacrifier le recall.
    """
    y_proba_train = model_rw.predict_proba(X_train_s)[:, 1]
    y_proba_test  = model_rw.predict_proba(X_test_s)[:, 1]

    adult_mask = jr_train == 0
    thr_adult = best_threshold_f1(y_train[adult_mask], y_proba_train[adult_mask])

    junior_mask = jr_train == 1
    thr_junior_strict = best_threshold_recall_target(
        y_train[junior_mask], y_proba_train[junior_mask], recall_target=0.55)

    y_pred = dual_threshold_predict(y_proba_test, jr_test, thr_adult, thr_junior_strict)
    return y_pred, thr_adult, thr_junior_strict

# ──────────────────────────────────────────────
# RAPPORT
# ──────────────────────────────────────────────
def write_report(lines, path):
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Rapport écrit : {path}")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    df = load_data()

    X_all = df[V5_FEATURES].fillna(0).values
    y_all = df[TARGET_COL].astype(int).values
    jr_all = df["is_junior"].values
    gender_all = df["gender"].fillna("Unknown").values

    (X_train, X_test,
     y_train, y_test,
     jr_train, jr_test,
     gender_train, gender_test) = train_test_split(
        X_all, y_all, jr_all, gender_all,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    df_test = pd.DataFrame({"is_junior": jr_test, "gender": gender_test})

    lines = [
        "RAPPORT FAIRNESS V2 — Corrections d'Équité",
        "=" * 60,
        f"Dataset test : {len(y_test)} CV  |  Positifs : {y_test.sum()}  |  Taux : {y_test.mean():.1%}",
        "",
    ]

    # ── BASELINE ──
    lines += ["── BASELINE (v5 — seuils fixes) ──────────────────────────────"]
    model_base, y_proba_base, y_pred_base, thr_a, thr_j = run_baseline(
        X_train_s, X_test_s, y_train, y_test, jr_test, df_test)

    lines += ["Par genre :"]
    for g in ["Female", "Male"]:
        mask = gender_test == g
        lines.append(group_metrics(y_test[mask], y_pred_base[mask], g))
    lines += ["Par âge :"]
    for is_jr, label in [(1, "Junior (<30)"), (0, "Adulte (30+)")]:
        mask = jr_test == is_jr
        lines.append(group_metrics(y_test[mask], y_pred_base[mask], label))
    lines += [f"  AUC global : {roc_auc_score(y_test, y_proba_base):.3f}", ""]

    # ── ÉTAPE 1 — REWEIGHTING ──
    lines += ["── ÉTAPE 1 : Reweighting Genre ────────────────────────────────"]
    model_rw, y_proba_rw, y_pred_rw, thr_a1, thr_j1 = run_step1_reweighting(
        X_train_s, X_test_s, y_train, y_test, jr_train, jr_test, gender_train)
    lines += [f"  Seuil adultes recalibré : {thr_a1:.3f}"]
    lines += [f"  Seuil juniors recalibré : {thr_j1:.3f}"]
    lines += ["Par genre :"]
    for g in ["Female", "Male"]:
        mask = gender_test == g
        lines.append(group_metrics(y_test[mask], y_pred_rw[mask], g))
    lines += ["Par âge :"]
    for is_jr, label in [(1, "Junior (<30)"), (0, "Adulte (30+)")]:
        mask = jr_test == is_jr
        lines.append(group_metrics(y_test[mask], y_pred_rw[mask], label))
    lines += [f"  AUC global : {roc_auc_score(y_test, y_proba_rw):.3f}", ""]

    # ── ÉTAPE 2 — SEUILS PAR GENRE ──
    lines += ["── ÉTAPE 2 : Seuils Différenciés Genre × Âge (Equalized Recall) ──"]
    y_pred_4way, thresholds_4 = run_step2_gender_thresholds(
        X_train_s, X_test_s, y_train, y_test,
        jr_train, jr_test, gender_train, gender_test, model_rw)
    for key, val in thresholds_4.items():
        lines.append(f"  Seuil {key[0]:6} × {key[1]:6} : {val:.3f}")
    lines += ["Par genre :"]
    for g in ["Female", "Male"]:
        mask = gender_test == g
        lines.append(group_metrics(y_test[mask], y_pred_4way[mask], g))
    lines += ["Par âge :"]
    for is_jr, label in [(1, "Junior (<30)"), (0, "Adulte (30+)")]:
        mask = jr_test == is_jr
        lines.append(group_metrics(y_test[mask], y_pred_4way[mask], label))
    lines += [f"  AUC global : {roc_auc_score(y_test, model_rw.predict_proba(X_test_s)[:, 1]):.3f}", ""]

    # ── ÉTAPE 3 — PRÉCISION JUNIORS ──
    lines += ["── ÉTAPE 3 : Amélioration Précision Juniors (recall_target=0.55) ──"]
    y_pred_step3, thr_a3, thr_j3 = run_step3_junior_precision(
        X_train_s, X_test_s, y_train, y_test, jr_train, jr_test, model_rw)
    lines += [f"  Seuil adultes : {thr_a3:.3f}  |  Seuil juniors : {thr_j3:.3f}"]
    lines += ["Par âge :"]
    for is_jr, label in [(1, "Junior (<30)"), (0, "Adulte (30+)")]:
        mask = jr_test == is_jr
        lines.append(group_metrics(y_test[mask], y_pred_step3[mask], label))
    lines += [""]

    # ── RÉSUMÉ COMPARATIF ──
    lines += [
        "── RÉSUMÉ COMPARATIF ─────────────────────────────────────────",
        f"{'Modèle':40} {'Female R':>10} {'Male R':>8} {'Jr R':>8} {'Jr P':>8}",
    ]
    configs = [
        ("Baseline v5", y_pred_base),
        ("+ Reweighting (étape 1)", y_pred_rw),
        ("+ Seuils genre×âge (étape 2)", y_pred_4way),
        ("+ Junior précision (étape 3)", y_pred_step3),
    ]
    for name, preds in configs:
        f_mask = gender_test == "Female"
        m_mask = gender_test == "Male"
        jr_mask = jr_test == 1
        fr = recall_score(y_test[f_mask], preds[f_mask], zero_division=0)
        mr = recall_score(y_test[m_mask], preds[m_mask], zero_division=0)
        jr = recall_score(y_test[jr_mask], preds[jr_mask], zero_division=0)
        jp = precision_score(y_test[jr_mask], preds[jr_mask], zero_division=0)
        lines.append(f"  {name:38} {fr:>10.3f} {mr:>8.3f} {jr:>8.3f} {jp:>8.3f}")

    write_report(lines, REPORTS_DIR / "fairness_v2.txt")
    print("Done.")

if __name__ == "__main__":
    main()
