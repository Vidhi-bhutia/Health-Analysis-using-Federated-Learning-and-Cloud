import os
import json
import pandas as pd
from xgboost import XGBClassifier

# Paths
BASE_DIR = "data/hospital/"
WEIGHT_DIR = "data/weights/stroke/"
os.makedirs(WEIGHT_DIR, exist_ok=True)

# Hospitals
HOSPITALS = ["Hospital A", "Hospital B", "Hospital C"]

for hospital in HOSPITALS:
	print(f"\n🏥 Training Stroke XGBoost (gblinear) Model for {hospital}...")

	# Load hospital-specific dataset
	DATA_PATH = os.path.join(BASE_DIR, hospital, "stroke.csv")
	df = pd.read_csv(DATA_PATH)

	# Features and Target
	X_df = df.drop(columns=["At Risk (Binary)"])
	y = df["At Risk (Binary)"].astype(int).values

	# Class imbalance handling
	pos = (y == 1).sum()
	neg = (y == 0).sum()
	scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

	# XGBoost linear booster tuned
	model = XGBClassifier(
		booster="gblinear",
		objective="binary:logistic",
		eval_metric="logloss",
		learning_rate=0.05,
		n_estimators=300,
		reg_alpha=0.0,
		reg_lambda=0.1,
		lambda_bias=0.0,
		updater="shotgun",
		scale_pos_weight=scale_pos_weight,
		random_state=42,
		n_jobs=-1,
	)
	model.fit(X_df.values, y)

	# Save weights.json compatible with FedAvg (nested)
	coef_list = model.coef_.ravel().tolist() if hasattr(model, "coef_") else [0.0] * X_df.shape[1]
	intercept_val = float(getattr(model, "intercept_", 0.0))
	weights = {
		"model": "xgboost_gblinear",
		"hospital": hospital,
		"features": X_df.columns.tolist(),
		"coef": [coef_list],
		"intercept": [intercept_val],
		"classes": [0, 1],
		"num_samples": int(len(y)),
	}
	file_path = os.path.join(WEIGHT_DIR, f"{hospital.lower().replace(' ', '_')}_weights.json")
	with open(file_path, "w") as f:
		json.dump(weights, f, indent=4)

	print(f"💾 Weights saved: {file_path}")
