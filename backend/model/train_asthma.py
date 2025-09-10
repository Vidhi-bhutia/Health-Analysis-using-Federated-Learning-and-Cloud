import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import json

# Paths
BASE_DIR = "data/hospital/"
WEIGHT_DIR = "data/weights/asthma/"
os.makedirs(WEIGHT_DIR, exist_ok=True)

# Hospitals
HOSPITALS = ["Hospital A", "Hospital B", "Hospital C"]

for hospital in HOSPITALS:
    print(f"\n🏥 Training Asthma Model for {hospital}...")

    # Load hospital-specific dataset
    DATA_PATH = os.path.join(BASE_DIR, hospital, "asthma.csv")
    df = pd.read_csv(DATA_PATH)

    # ---- Create Binary Target ----
    df["Asthma"] = df[["Severity_Mild", "Severity_Moderate"]].max(axis=1)

    # Drop severity columns
    X = df.drop(columns=["Severity_Mild", "Severity_Moderate", "Severity_None", "Asthma"])
    y = df["Asthma"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression
    model = LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced")
    model.fit(X_train, y_train)

    # Save Weights in Consistent Format
    weights = {
        "model": "logistic_regression",
        "hospital": hospital,
        "features": X.columns.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "classes": model.classes_.tolist()
    }

    file_path = os.path.join(WEIGHT_DIR, f"{hospital.lower().replace(' ', '_')}_weights.json")
    with open(file_path, "w") as f:
        json.dump(weights, f, indent=4)

    print(f"💾 Weights saved: {file_path}")
