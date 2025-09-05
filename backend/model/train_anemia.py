import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# Paths
ROOT = Path(__file__).resolve().parents[2]
HOSP_DIR = ROOT / "data" / "hospital"
WEIGHTS_DIR = ROOT / "data" / "weights" / "anemia"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

HOSPITALS = ["Hospital A", "Hospital B", "Hospital C"]

def train_anemia():
    for hosp in HOSPITALS:
        csv_path = HOSP_DIR / hosp / "anemia.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing file: {csv_path}")
            continue

        print(f"Training Anemia model for {hosp}...")

        # Load dataset
        df = pd.read_csv(csv_path)

        # Features and target
        X = df.drop(columns=["Result"]).values
        y = df["Result"].astype(int).values

        # Logistic Regression
        model = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=500,
            random_state=42
        )
        model.fit(X, y)

        # Save weights only
        weights = {
            "model": "logistic_regression",
            "hospital": hosp,
            "features": df.drop(columns=["Result"]).columns.tolist(),
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "classes": model.classes_.tolist()
        }

        out_path = WEIGHTS_DIR / f"{hosp.replace(' ', '_').lower()}_weights.json"
        with open(out_path, "w") as f:
            json.dump(weights, f, indent=2)

        print(f"✅ Weights saved to {out_path}")

if __name__ == "__main__":
    train_anemia()
