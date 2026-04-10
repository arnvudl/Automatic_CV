"""
run.py — Interface de lancement du pipeline ML (Reorganized)
"""

import sys
import time
import subprocess
from pathlib import Path

# Ajouter la racine au path pour les imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline_ml.core import parse, features, train, audit, tune

SEP = "=" * 55

def run_dashboard():
    print(f"\nLancement du dashboard dynamique...")
    print(f"L'application sera disponible sur http://127.0.0.1:8050")
    try:
        subprocess.run([sys.executable, str(ROOT / "pipeline_ml" / "dashboard" / "app.py")])
    except KeyboardInterrupt:
        print("\nDashboard arrêté.")

def header():
    print(f"\n{SEP}")
    print("  CV-Intelligence — Pipeline ML")
    print("  TechCore Liege")
    print(SEP)

def run_full():
    header()
    print("\nLancement du pipeline complet...\n")
    steps = [
        (parse.main,    "Parsing CV"),
        (features.main, "Feature engineering"),
        (tune.main,     "Optimisation hyperparamètres + seuil"),
        (audit.main,    "Audit biais + SHAP"),
    ]
    for i, (fn, label) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {label}...")
        fn()
    print("\nPipeline terminé.")

def menu():
    header()
    options = {
        "1": ("Parsing CV (Raw -> Features/Identities)", parse.main),
        "2": ("Feature Engineering (Enrichissement)",    features.main),
        "3": ("Entraînement Standard",                  train.main),
        "4": ("Optimisation (Tuning + Seuil Opti)",     tune.main),
        "5": ("Audit (Biais & Fairlearn)",              audit.main),
        "6": ("Lancer le Dashboard Dynamique",          run_dashboard),
        "7": ("Pipeline COMPLET (Parsing -> Audit)",    run_full),
        "q": ("Quitter",                                 None),
    }

    while True:
        print("\nMenu Principal:")
        for k, (label, _) in options.items():
            print(f"  [{k}] {label}")
        
        choice = input("\nChoix : ").strip().lower()
        if choice == "q": break
        if choice == "7": run_full()
        elif choice == "6": run_dashboard()
        elif choice in options:
            label, fn = options[choice]
            if fn: fn()
        else: print("Choix invalide.")

if __name__ == "__main__":
    menu()
