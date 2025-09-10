import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import os
import json

# Paths
BASE_DIR = "data/hospital/"
WEIGHT_DIR = "data/weights/diabetes/"
os.makedirs(WEIGHT_DIR, exist_ok=True)

# Hospitals
HOSPITALS = ["Hospital A", "Hospital B", "Hospital C"]

for hospital in HOSPITALS:
    print(f"\n🏥 Training Diabetes Model for {hospital}...")

    # Load hospital-specific dataset
    DATA_PATH = os.path.join(BASE_DIR, hospital, "diabetes.csv")
    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    X = df.drop(columns=["diabetes"])
    y = df["diabetes"]

    # One-hot encode categorical columns
    cat_cols = ["gender", "smoking_history"]
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression
    model = LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced")
    model.fit(X_train, y_train)

    # Save weights in consistent JSON format
    weights = {
        "model": "logistic_regression",
        "hospital": hospital,
        "features": X_encoded.columns.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "classes": model.classes_.tolist()
    }

    file_path = os.path.join(WEIGHT_DIR, f"{hospital.lower().replace(' ', '_')}_weights.json")
    with open(file_path, "w") as f:
        json.dump(weights, f, indent=4)

    print(f"💾 Weights saved: {file_path}")
