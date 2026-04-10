"""
run.py — Interface de lancement du pipeline ML CV-Intelligence

Usage :
  python pipeline_ml/run.py          -> menu interactif
  python pipeline_ml/run.py full     -> pipeline complet sans prompt
"""

import sys
import time
from pathlib import Path

# Ajouter la racine au path pour les imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline_ml import parse_cv, feature_engineering, train, audit

SEP = "=" * 55

def header():
    print(f"\n{SEP}")
    print("  CV-Intelligence — Pipeline ML")
    print("  TechCore Liege")
    print(SEP)

def step(n: int, total: int, title: str):
    print(f"\n[{n}/{total}] {title}")
    print("-" * 40)

def run_step(fn) -> bool:
    t0 = time.time()
    try:
        fn()
        print(f"  OK ({time.time() - t0:.1f}s)")
        return True
    except Exception as e:
        print(f"  ERREUR : {e}")
        return False

def run_full():
    header()
    print("\nLancement du pipeline complet...\n")

    steps = [
        (parse_cv.main,            "Parsing + pseudonymisation (labels reels)"),
        (feature_engineering.main, "Feature engineering"),
        (train.main,               "Entrainement du modele"),
        (audit.main,               "Audit biais + SHAP"),
    ]

    ok = 0
    for i, (fn, label) in enumerate(steps, 1):
        step(i, len(steps), label)
        if run_step(fn):
            ok += 1
        else:
            print(f"\nPipeline arrete a l'etape {i}. Corrige l'erreur et relance.")
            return

    print(f"\n{SEP}")
    print(f"  Pipeline termine : {ok}/{len(steps)} etapes reussies")
    print(f"  features.csv  -> data/processed/features.csv")
    print(f"  Modele        -> models/model.pkl")
    print(f"  Audit         -> reports/audit.txt")
    print(SEP)

def menu():
    header()

    options = {
        "1": ("Parsing + pseudonymisation (labels reels)", parse_cv.main),
        "2": ("Feature engineering",                       feature_engineering.main),
        "3": ("Entrainement du modele",                    train.main),
        "4": ("Audit biais + SHAP",                        audit.main),
        "5": ("Pipeline complet (1->4)",                   None),
        "q": ("Quitter",                                   None),
    }

    while True:
        print("\nQue voulez-vous lancer ?")
        print()
        for key, (label, _) in options.items():
            print(f"  [{key}] {label}")
        print()

        choice = input("Choix : ").strip().lower()

        if choice == "q":
            print("Bye.")
            break
        elif choice == "5":
            run_full()
        elif choice in options:
            label, fn = options[choice]
            step(int(choice), 3, label)
            run_step(fn)
        else:
            print("  Choix invalide.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        run_full()
    else:
        menu()
