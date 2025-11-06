import pandas as pd
import numpy as np
import json
from pathlib import Path
from xgboost import XGBClassifier
from joblib import dump

# Paths
ROOT = Path(__file__).resolve().parents[2]
HOSP_DIR = ROOT / "data" / "hospital"
WEIGHTS_DIR = ROOT / "data" / "weights" / "anemia"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

HOSPITALS = ["Hospital A", "Hospital B", "Hospital C"]


def _logit(p: float) -> float:
	p = min(max(p, 1e-6), 1 - 1e-6)
	return float(np.log(p / (1 - p)))


def train_anemia():
	for hosp in HOSPITALS:
		csv_path = HOSP_DIR / hosp / "anemia.csv"
		if not csv_path.exists():
			print(f"[WARN] Missing file: {csv_path}")
			continue

		print(f"Training Anemia XGBoost (gblinear) model for {hosp}...")

		# Load dataset
		df = pd.read_csv(csv_path)

		# Features and target
		X_df = df.drop(columns=["Result"])
		y = df["Result"].astype(int).values

		n_pos = int((y == 1).sum())
		n_neg = int((y == 0).sum())
		n_total = int(len(y))

		# Handle single-class case: save bias-only model
		if len(np.unique(y)) < 2:
			p = float(y.mean())
			intercept_val = _logit(p)
			coef_list = [0.0] * X_df.shape[1]
			weights = {
				"model": "xgboost_gblinear",
				"hospital": hosp,
				"features": X_df.columns.tolist(),
				"coef": [coef_list],
				"intercept": [intercept_val],
				"classes": [0, 1],
				"num_samples": n_total,
				"single_class": True,
			}
			out_path = WEIGHTS_DIR / f"{hosp.replace(' ', '_').lower()}_weights.json"
			with open(out_path, "w") as f:
				json.dump(weights, f, indent=2)
			print(f"✅ Saved bias-only weights (single-class) to {out_path}")
			continue

		# Class imbalance handling
		scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

		# XGBoost linear booster with tuned regularization
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

		# Save weights in LR-compatible nested format for FedAvg
		coef_list = model.coef_.ravel().tolist() if hasattr(model, "coef_") else [0.0] * X_df.shape[1]
		intercept_val = float(getattr(model, "intercept_", 0.0))
		weights = {
			"model": "xgboost_gblinear",
			"hospital": hosp,
			"features": X_df.columns.tolist(),
			"coef": [coef_list],
			"intercept": [intercept_val],
			"classes": [0, 1],
			"num_samples": n_total,
		}
		out_path = WEIGHTS_DIR / f"{hosp.replace(' ', '_').lower()}_weights.json"
		with open(out_path, "w") as f:
			json.dump(weights, f, indent=2)

		print(f"✅ Weights saved to {out_path}")


if __name__ == "__main__":
	train_anemia()
