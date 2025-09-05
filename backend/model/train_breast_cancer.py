import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import json

# Paths
BASE_DIR = "data/hospital/"
WEIGHT_DIR = "data/weights/breast_cancer/"
os.makedirs(WEIGHT_DIR, exist_ok=True)

# Hospitals
HOSPITALS = ["Hospital A", "Hospital B", "Hospital C"]

for hospital in HOSPITALS:
    print(f"\n🏥 Training Breast Cancer Model for {hospital}...")

    # Load hospital-specific dataset
    DATA_PATH = os.path.join(BASE_DIR, hospital, "breast_cancer.csv")
    df = pd.read_csv(DATA_PATH)

    # Features and Target
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression
    model = LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced")
    model.fit(X_train, y_train)

    # Save Weights (Consistent JSON Format)
    weights = {
        "model": "logistic_regression",
        "hospital": hospital,
        "features": X.columns.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "classes": model.classes_.tolist()
    }

    file_path = os.path.join(WEIGHT_DIR, f"{hospital.lower().replace(' ', '_')}_breast_cancer.json")
    with open(file_path, "w") as f:
        json.dump(weights, f, indent=4)

    print(f"💾 Weights saved: {file_path}")
