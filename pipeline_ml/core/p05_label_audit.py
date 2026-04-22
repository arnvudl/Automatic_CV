"""
p05_label_audit.py — Audit de biais dans les labels bruts
==========================================================
Objectif : mesurer AVANT tout modèle ML si les labels (invite/reject)
sont distribués de façon équitable entre les groupes démographiques.

Si les labels sont biaisés, le modèle apprendra ce biais.
Ce script produit des preuves factuelles, sans interprétation orientée.

Sorties : console + reports/label_audit.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT            = Path(__file__).parent.parent.parent
FEATURES_PATH   = ROOT / "data" / "processed" / "features.csv"
IDENTITIES_PATH = ROOT / "data" / "processed" / "identities.csv"
REPORTS_DIR     = ROOT / "reports"

EDU_LABELS  = {1: "Bac ou moins", 2: "Bachelor", 3: "Master", 4: "PhD"}
AGE_BINS    = [0, 29, 45, 99]
AGE_LABELS  = ["Junior (<30)", "Adulte (30-45)", "Senior (>45)"]
SIG_LEVELS  = {0.001: "***", 0.01: "**", 0.05: "*", 1.0: "ns"}


# ── Helpers ────────────────────────────────────────────────────────

def sig_stars(p: float) -> str:
    for threshold, stars in SIG_LEVELS.items():
        if p < threshold:
            return stars
    return "ns"


def invite_rate_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Taux d'invitation et test chi² par groupe."""
    rows = []
    global_rate = df["label"].mean()

    for grp, sub in df.groupby(group_col):
        n       = len(sub)
        invited = sub["label"].sum()
        rate    = invited / n if n > 0 else 0.0

        # Chi² : ce groupe vs tous les autres
        others  = df[df[group_col] != grp]
        table   = np.array([
            [invited,          n - invited],
            [others["label"].sum(), len(others) - others["label"].sum()],
        ])
        if table.min() >= 5:
            chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
        else:
            _, p = stats.fisher_exact(table)
            chi2 = None

        delta = rate - global_rate
        rows.append({
            "Groupe":     str(grp),
            "N":          n,
            "Invités":    int(invited),
            "Taux (%)":   round(rate * 100, 1),
            "Δ global":   f"{delta:+.1%}",
            "p-value":    round(p, 4),
            "Sig.":       sig_stars(p),
            "Direction":  "↑ favorisé" if delta > 0.05 else ("↓ défavorisé" if delta < -0.05 else "≈ neutre"),
        })

    return pd.DataFrame(rows).sort_values("Taux (%)", ascending=False)


def feature_means_by_label(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Moyenne des features par label avec test Mann-Whitney U."""
    rows = []
    invited  = df[df["label"] == 1]
    rejected = df[df["label"] == 0]

    for feat in features:
        col = pd.to_numeric(df[feat], errors="coerce")
        a   = pd.to_numeric(invited[feat],  errors="coerce").dropna()
        b   = pd.to_numeric(rejected[feat], errors="coerce").dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({
            "Feature":          feat,
            "Moy. invités":     round(a.mean(), 3),
            "Moy. rejetés":     round(b.mean(), 3),
            "Δ":                round(a.mean() - b.mean(), 3),
            "p-value":          round(p, 4),
            "Sig.":             sig_stars(p),
            "Biais potentiel":  "OUI" if p < 0.05 and abs(a.mean() - b.mean()) / (col.std() + 1e-9) > 0.3 else "",
        })

    return pd.DataFrame(rows).sort_values("p-value")


def section(title: str, out_lines: list) -> None:
    line = f"\n{'=' * 60}\n  {title}\n{'=' * 60}"
    _print(line)
    out_lines.append(line)


def _print(text: str) -> None:
    """Print robuste — ignore les caractères non encodables sur Windows."""
    print(text.encode("cp1252", errors="replace").decode("cp1252"))


def print_and_save(text: str, out_lines: list) -> None:
    _print(text)
    out_lines.append(text)


# ── Main ───────────────────────────────────────────────────────────

def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    out_lines = []

    if not FEATURES_PATH.exists():
        print("features.csv introuvable — lancez p01 + p02 d'abord.")
        return

    df      = pd.read_csv(FEATURES_PATH)
    df_id   = pd.read_csv(IDENTITIES_PATH) if IDENTITIES_PATH.exists() else pd.DataFrame()

    target = "label" if "label" in df.columns else "passed_next_stage"
    df = df[df[target].notna()].copy()
    df["label"] = df[target].astype(int)

    id_cols = ["cv_id", "gender", "age"] + [c for c in ["city", "country"] if c in df_id.columns]
    if not df_id.empty:
        df = df.merge(df_id[id_cols], on="cv_id", how="left")

    n_total   = len(df)
    n_invited = int(df["label"].sum())
    global_rate = df["label"].mean()

    header = (
        f"\nAUDIT DE BIAIS — LABELS BRUTS\n"
        f"{'─' * 60}\n"
        f"Dataset : {n_total} CV  |  Invités : {n_invited} ({global_rate:.1%})  "
        f"|  Rejetés : {n_total - n_invited} ({1 - global_rate:.1%})\n"
        f"{'─' * 60}\n"
        f"Lecture : * p<0.05  ** p<0.01  *** p<0.001  ns = non significatif\n"
        f"Δ global = écart au taux moyen d'invitation ({global_rate:.1%})\n"
    )
    print_and_save(header, out_lines)

    # ── 1. Genre ───────────────────────────────────────────────────
    if "gender" in df.columns:
        section("1. BIAIS DE GENRE", out_lines)
        df_g = df[df["gender"].isin(["Male", "Female"])].copy()
        tbl  = invite_rate_by_group(df_g, "gender")
        print_and_save(tbl.to_string(index=False), out_lines)

    # ── 2. Niveau d'éducation ──────────────────────────────────────
    section("2. BIAIS DE NIVEAU D'ÉDUCATION", out_lines)
    df["edu_label"] = pd.to_numeric(df["education_level"], errors="coerce")\
                        .round().astype("Int64").map(EDU_LABELS)
    tbl = invite_rate_by_group(df[df["edu_label"].notna()], "edu_label")
    print_and_save(tbl.to_string(index=False), out_lines)

    note = (
        "\n  → Un écart significatif ici signifie que les recruteurs humains\n"
        "    ont historiquement favorisé certains diplômes dans leurs décisions.\n"
        "    Ce biais est encodé dans les labels et sera appris par le modèle."
    )
    print_and_save(note, out_lines)

    # ── 3. Groupe d'âge ───────────────────────────────────────────
    if "age" in df.columns:
        section("3. BIAIS D'ÂGE", out_lines)
        df["age_num"] = pd.to_numeric(df["age"], errors="coerce")
        df["age_group"] = pd.cut(df["age_num"], bins=AGE_BINS, labels=AGE_LABELS)
        df_a = df[df["age_group"].notna()].copy()
        tbl  = invite_rate_by_group(df_a, "age_group")
        print_and_save(tbl.to_string(index=False), out_lines)

    # ── 4. Pays ────────────────────────────────────────────────────
    if "country" in df.columns and df["country"].notna().sum() > 0:
        section("4. BIAIS GEOGRAPHIQUE (pays de residence)", out_lines)
        df_c = df[df["country"].notna()].copy()
        tbl  = invite_rate_by_group(df_c, "country")
        print_and_save(tbl.to_string(index=False), out_lines)
        note_geo = (
            "\n  -> Attention : le pays de residence n'est PAS utilise comme feature ML\n"
            "     (ce serait discriminatoire). Il est ici uniquement pour l'audit des labels."
        )
        print_and_save(note_geo, out_lines)

    # ── 5. Secteur ─────────────────────────────────────────────────
    section("5. BIAIS DE SECTEUR", out_lines)
    tbl = invite_rate_by_group(df[df["sector"].notna()], "sector")
    print_and_save(tbl.to_string(index=False), out_lines)

    # ── 6. Features : différence invités vs rejetés ────────────────
    section("6. FEATURES — DIFFERENCE INVITES vs REJETES (Mann-Whitney U)", out_lines)
    feat_cols = [
        "years_experience", "avg_job_duration", "education_level",
        "nb_jobs", "nb_technical_skills", "nb_methods_skills",
        "nb_certifications", "nb_languages", "english_level",
        "career_progression", "total_skills",
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]
    tbl = feature_means_by_label(df, feat_cols)
    print_and_save(tbl.to_string(index=False), out_lines)

    note2 = (
        "\n  → Les features marquées 'OUI' (Biais potentiel) ont une différence\n"
        "    statistiquement significative ET pratiquement importante entre\n"
        "    invités et rejetés. Elles reflètent les critères des recruteurs humains.\n"
        "    Si ces critères sont eux-mêmes biaisés, le modèle amplifiera ce biais."
    )
    print_and_save(note2, out_lines)

    # ── 7. Croisements (éducation × genre) ────────────────────────
    if "gender" in df.columns:
        section("7. CROISEMENT EDUCATION x GENRE", out_lines)
        df_cross = df[df["gender"].isin(["Male", "Female"]) & df["edu_label"].notna()].copy()
        cross = df_cross.groupby(["edu_label", "gender"])["label"].agg(
            N="count", Invités="sum"
        ).reset_index()
        cross["Taux (%)"] = (cross["Invités"] / cross["N"] * 100).round(1)
        cross = cross.sort_values(["edu_label", "gender"])
        print_and_save(cross.to_string(index=False), out_lines)

    # ── 8. Synthèse ────────────────────────────────────────────────
    section("8. SYNTHESE — BIAIS CONFIRMES DANS LES LABELS", out_lines)

    findings = []

    # Éducation — toujours tester
    edu_rates = df.groupby("edu_label")["label"].mean()
    if not edu_rates.empty:
        max_edu = edu_rates.idxmax()
        min_edu = edu_rates.idxmin()
        gap_edu = edu_rates.max() - edu_rates.min()
        if gap_edu > 0.1:
            findings.append(
                f"  ● Biais académique : {max_edu} ({edu_rates.max():.1%}) vs "
                f"{min_edu} ({edu_rates.min():.1%}) — écart {gap_edu:.1%}"
            )

    # Genre
    if "gender" in df.columns:
        g_rates = df[df["gender"].isin(["Male","Female"])].groupby("gender")["label"].mean()
        if len(g_rates) == 2:
            gap_g = abs(g_rates["Male"] - g_rates["Female"])
            direction = "hommes favorisés" if g_rates["Male"] > g_rates["Female"] else "femmes favorisées"
            findings.append(
                f"  ● Biais de genre : {direction} — écart {gap_g:.1%} "
                f"(M={g_rates['Male']:.1%}, F={g_rates['Female']:.1%})"
            )

    # Âge
    if "age_group" in df.columns:
        a_rates = df.groupby("age_group", observed=True)["label"].mean()
        if not a_rates.empty:
            gap_a = a_rates.max() - a_rates.min()
            if gap_a > 0.1:
                findings.append(
                    f"  ● Biais d'âge : {a_rates.idxmax()} favorisé "
                    f"({a_rates.max():.1%}) vs {a_rates.idxmin()} ({a_rates.min():.1%})"
                )

    if findings:
        print_and_save("Biais statistiquement détectés dans les labels :\n", out_lines)
        for f in findings:
            print_and_save(f, out_lines)
        conclusion = (
            "\n  ⚠ Ces biais sont dans les DONNÉES, pas dans le modèle.\n"
            "    Un modèle entraîné sur ces labels les reproduira fidèlement.\n"
            "    La seule correction radicale est le relabeling humain des cas litigieux."
        )
    else:
        conclusion = "\n  ✓ Aucun biais majeur détecté dans les labels (seuil > 10%)."

    print_and_save(conclusion, out_lines)

    # ── Sauvegarde ─────────────────────────────────────────────────
    report_path = REPORTS_DIR / "label_audit.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"\nRapport sauvegardé : {report_path}")


if __name__ == "__main__":
    main()
