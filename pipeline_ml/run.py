"""
run.py — Pipeline ML Runner v3
Etapes : p00 exploration → p01 parse → p02 features → p03 analyse → p04 train → p06 audit
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline_ml.core import p00_exploration as p0
from pipeline_ml.core import p01_parse       as p1
from pipeline_ml.core import p02_features    as p2
from pipeline_ml.core import p03_analysis    as p3
from pipeline_ml.core import p04_train       as p4
from pipeline_ml.core import p06_audit       as p6

SEP = "=" * 55


def run_dashboard():
    print("\nLancement du dashboard...")
    print("Disponible sur http://127.0.0.1:8050")
    try:
        subprocess.run([sys.executable, str(ROOT / "pipeline_ml" / "dashboard" / "app.py")])
    except KeyboardInterrupt:
        print("\nDashboard arrete.")


def header():
    print(f"\n{SEP}")
    print("  CV-Intelligence — Pipeline ML v3")
    print("  TechCore Liege")
    print(SEP)


def run_full():
    header()
    print("\nLancement du pipeline complet...\n")
    steps = [
        (p0.main, "00 - Exploration des donnees brutes"),
        (p1.main, "01 - Parsing CV"),
        (p2.main, "02 - Feature Engineering v3"),
        (p3.main, "03 - EDA & Analyse Statistique"),
        (p4.main, "04 - Entrainement (Fairness-Aware)"),
        (p6.main, "06 - Audit Biais, Equite & SHAP"),
    ]
    for i, (fn, label) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {label}...")
        fn()
    print("\nPipeline termine.")


def menu():
    header()
    options = {
        "0": ("00 - Exploration des donnees brutes",     p0.main),
        "1": ("01 - Parsing CV (Raw -> Features)",       p1.main),
        "2": ("02 - Feature Engineering v3",             p2.main),
        "3": ("03 - EDA & Analyse Statistique",          p3.main),
        "4": ("04 - Entrainement Fairness-Aware",        p4.main),
        "6": ("06 - Audit Biais, Equite & SHAP",        p6.main),
        "7": ("07 - Dashboard Dynamique",                run_dashboard),
        "9": ("09 - Pipeline COMPLET (00 → 06)",         run_full),
        "q": ("Quitter",                                 None),
    }

    while True:
        print("\nMenu Principal:")
        for k, (label, _) in options.items():
            print(f"  [{k}] {label}")
        choice = input("\nChoix : ").strip().lower()
        if choice == "q":
            break
        elif choice == "9":
            run_full()
        elif choice in options:
            label, fn = options[choice]
            if fn:
                fn()
        else:
            print("Choix invalide.")


if __name__ == "__main__":
    menu()
