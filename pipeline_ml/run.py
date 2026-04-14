"""
run.py - Pipeline ML Runner
"""

import sys
import subprocess
from pathlib import Path

# Add root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline_ml.core import p00_exploration as p0
from pipeline_ml.core import p01_parse as p1
from pipeline_ml.core import p02_features as f2
from pipeline_ml.core import p03_analysis as a3
from pipeline_ml.core import p04_train as t4
from pipeline_ml.core import p05_tune as t5
from pipeline_ml.core import p06_audit as a6
from pipeline_ml.core import p07_labeling as l7

SEP = "=" * 55

def run_dashboard():
    print("\nLancement du dashboard dynamique...")
    print("L'application sera disponible sur http://127.0.0.1:8050")
    try:
        subprocess.run([sys.executable, str(ROOT / "pipeline_ml" / "dashboard" / "app.py")])
    except KeyboardInterrupt:
        print("\nDashboard arrete.")

def header():
    print(f"\n{SEP}")
    print("  CV-Intelligence - Pipeline ML")
    print("  TechCore Liege")
    print(SEP)

def run_full():
    header()
    print("\nLancement du pipeline complet...\n")
    steps = [
        (p0.main,  "00 - Exploration des données brutes"),
        (p1.main,  "01 - Parsing CV"),
        (f2.main,  "02 - Feature engineering"),
        (a3.main,  "03 - Exploration & Stat (EDA)"),
        (t4.main,  "04 - Entrainement Standard"),
        (t5.main,  "05 - Optimisation (Tuning + Seuil)"),
        (a6.main,  "06 - Audit Biais + SHAP"),
        (l7.main,  "07 - Labeling (Pseudo-labels)"),
    ]
    for i, (fn, label) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {label}...")
        fn()
    print("\nPipeline termine.")

def menu():
    header()
    options = {
        "0": ("00 - Exploration des données brutes (EDA)", p0.main),
        "1": ("01 - Parsing CV (Raw -> Features/Identities)", p1.main),
        "2": ("02 - Feature Engineering (Enrichissement)", f2.main),
        "3": ("03 - Exploration & Analyse Statistique (EDA)", a3.main),
        "4": ("04 - Entrainement Standard", t4.main),
        "5": ("05 - Optimisation (Tuning + Seuil Opti)", t5.main),
        "6": ("06 - Audit (Biais & Fairlearn)", a6.main),
        "7": ("07 - Labeling (Pseudo-labels)", l7.main),
        "8": ("08 - Lancer le Dashboard Dynamique", run_dashboard),
        "9": ("09 - Pipeline COMPLET (00 -> 07)", run_full),
        "q": ("Quitter", None),
    }

    while True:
        print("\nMenu Principal:")
        for k, (label, _) in options.items():
            print(f"  [{k}] {label}")
        
        choice = input("\nChoix : ").strip().lower()
        if choice == "q": break
        if choice == "9": run_full()
        elif choice == "8": run_dashboard()
        elif choice in options:
            label, fn = options[choice]
            if fn: fn()
        else: print("Choix invalide.")

if __name__ == "__main__":
    menu()
