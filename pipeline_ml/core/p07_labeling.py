"""
labeling.py — Pseudo-labels (Refactored)
"""

import pandas as pd
from pathlib import Path

# ==============================================================
# CONFIG (Adaptée pour core/labeling.py)
# ==============================================================
ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "processed" / "features.csv"

def main():
    if not DATA_PATH.exists():
        print(f"Dataset introuvable : {DATA_PATH}")
        return
    df = pd.read_csv(DATA_PATH)
    # Logique simplifiée : les labels sont déjà dans dataset.csv via student_labels.csv
    print(f"Vérification des labels sur {len(df)} lignes.")

if __name__ == "__main__":
    main()
