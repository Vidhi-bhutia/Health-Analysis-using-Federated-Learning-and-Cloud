# Federated Weights Evaluation (Pre-FedAvg vs Post-FedAvg)
# - Computes accuracy, precision, recall, F1, AUC, log loss, specificity
# - Combines all confusion matrices into a single figure per disease
# - Adds metrics line chart and ROC curves in the same figure
# - Saves one figure per disease + CSV summary (minimal images)

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	roc_auc_score,
	roc_curve,
	precision_score,
	recall_score,
	f1_score,
	log_loss,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = DATA_DIR / "weights"
RAW_DIR = DATA_DIR / "raw"
FIG_DIR = ROOT / "static" / "img" / "eval"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HOSPITAL_SLUGS = ["hospital_a", "hospital_b", "hospital_c"]
DISEASES = {
	"anemia": {
		"raw_file": "anemia.csv",
		"target": "Result",
		"preprocess": None,
	},
	"breast_cancer": {
		"raw_file": "breast_cancer.csv",
		"target": "diagnosis",
		"preprocess": None,
	},
	"diabetes": {
		"raw_file": "diabetes.csv",
		"target": "diabetes",
		"preprocess": "diabetes_onehot",
	},
	"stroke": {
		"raw_file": "stroke.csv",
		"target": "At Risk (Binary)",
		"preprocess": None,
	},
}

def load_weights(disease: str, hospital_slug: str) -> Dict:
	path = WEIGHTS_DIR / disease / f"{hospital_slug}_weights.json"
	if not path.exists():
		return {}
	with open(path, "r") as f:
		return json.load(f)


def extract_coef_intercept(weights: Dict) -> Tuple[np.ndarray, float, List[str]]:
	features = weights.get("features", [])
	raw_coef = weights.get("coef", [])
	if isinstance(raw_coef, list) and raw_coef and isinstance(raw_coef[0], list):
		coef = np.array(raw_coef[0], dtype=float)
	else:
		coef = np.array(raw_coef, dtype=float)
	raw_intercept = weights.get("intercept", 0.0)
	if isinstance(raw_intercept, list):
		intercept = float(raw_intercept[0] if raw_intercept else 0.0)
	else:
		intercept = float(raw_intercept)
	return coef, intercept, features


def fedavg_weights(weights_list: List[Dict]) -> Dict:
	valid = [w for w in weights_list if w]
	if not valid:
		return {}
	base_features = valid[0].get("features", [])
	coef_sum = np.zeros(len(base_features), dtype=float)
	intercept_sum = 0.0
	total_weight = 0.0
	use_weighted = any("num_samples" in w for w in valid)
	for w in valid:
		coef, intercept, feats = extract_coef_intercept(w)
		if feats != base_features or coef.shape[0] != len(base_features):
			continue
		weight = float(w.get("num_samples", 1.0)) if use_weighted else 1.0
		coef_sum += weight * coef
		intercept_sum += weight * intercept
		total_weight += weight
	if total_weight == 0:
		return {}
	coef_avg = (coef_sum / total_weight).tolist()
	intercept_avg = float(intercept_sum / total_weight)
	return {
		"features": base_features,
		"coef": [coef_avg],
		"intercept": [intercept_avg],
		"classes": valid[0].get("classes", [0,1])
	}


def sigmoid(z: np.ndarray) -> np.ndarray:
	z_clip = np.clip(z, -700, 700)
	return 1.0 / (1.0 + np.exp(-z_clip))


def predict_from_weights(weights: Dict, X_df: pd.DataFrame) -> np.ndarray:
	coef, intercept, feats = extract_coef_intercept(weights)
	for f in feats:
		if f not in X_df.columns:
			X_df[f] = 0.0
	X_aligned = X_df[feats].astype(float).values
	z = X_aligned @ coef + intercept
	p = sigmoid(z)
	return p


def preprocess_dataset(disease: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
	info = DISEASES[disease]
	target = info["target"]
	if info["preprocess"] == "diabetes_onehot":
		y = df[target].astype(int).values
		X = df.drop(columns=[target])
		X_df = pd.get_dummies(X, columns=["gender", "smoking_history"], drop_first=False)
		return X_df, y
	y = df[target].astype(int).values
	X_df = df.drop(columns=[target])
	return X_df, y


def _specificity(cm: np.ndarray) -> float:
	tn, fp, fn, tp = cm.ravel()
	den = (tn + fp)
	return float(tn / den) if den > 0 else np.nan


def evaluate_disease(disease: str) -> pd.DataFrame:
	print(f"\n=== Evaluating {disease.upper()} ===")
	raw_path = RAW_DIR / DISEASES[disease]["raw_file"]
	df = pd.read_csv(raw_path)
	X_df, y = preprocess_dataset(disease, df.copy())

	w_list = [load_weights(disease, h) for h in HOSPITAL_SLUGS]

	models = []
	probs = {}
	metrics = []

	for h, w in zip(HOSPITAL_SLUGS, w_list):
		if not w:
			continue
		p = predict_from_weights(w, X_df.copy())
		y_pred = (p >= 0.5).astype(int)
		cm = confusion_matrix(y, y_pred)
		acc = accuracy_score(y, y_pred)
		prec = precision_score(y, y_pred, zero_division=0)
		rec = recall_score(y, y_pred, zero_division=0)
		f1 = f1_score(y, y_pred, zero_division=0)
		try:
			auc = roc_auc_score(y, p)
		except Exception:
			auc = np.nan
		try:
			ll = log_loss(y, p, eps=1e-12)
		except Exception:
			ll = np.nan
		spec = _specificity(cm)
		models.append(h)
		probs[h] = p
		metrics.append({
			"disease": disease,
			"model": h,
			"kind": "pre_fedavg",
			"accuracy": acc,
			"precision": prec,
			"recall": rec,
			"f1": f1,
			"auc": auc,
			"log_loss": ll,
			"specificity": spec,
			"cm": cm,
		})

	w_avg = fedavg_weights(w_list)
	if w_avg:
		p = predict_from_weights(w_avg, X_df.copy())
		y_pred = (p >= 0.5).astype(int)
		cm = confusion_matrix(y, y_pred)
		acc = accuracy_score(y, y_pred)
		prec = precision_score(y, y_pred, zero_division=0)
		rec = recall_score(y, y_pred, zero_division=0)
		f1 = f1_score(y, y_pred, zero_division=0)
		try:
			auc = roc_auc_score(y, p)
		except Exception:
			auc = np.nan
		try:
			ll = log_loss(y, p, eps=1e-12)
		except Exception:
			ll = np.nan
		spec = _specificity(cm)
		models.append("fedavg")
		probs["fedavg"] = p
		metrics.append({
			"disease": disease,
			"model": "fedavg",
			"kind": "post_fedavg",
			"accuracy": acc,
			"precision": prec,
			"recall": rec,
			"f1": f1,
			"auc": auc,
			"log_loss": ll,
			"specificity": spec,
			"cm": cm,
		})

	if metrics:
		res_df = pd.DataFrame(metrics)
		ordered = [m for m in ["hospital_a", "hospital_b", "hospital_c", "fedavg"] if m in res_df["model"].tolist()]
		res_df["model"] = pd.Categorical(res_df["model"], categories=ordered, ordered=True)
		res_df = res_df.sort_values("model")

		fig = plt.figure(figsize=(18, 8))
		gs = GridSpec(2, 4, figure=fig, height_ratios=[2, 1.6])
		for idx, m in enumerate(ordered[:4]):
			ax = fig.add_subplot(gs[0, idx])
			cm = res_df.loc[res_df["model"] == m, "cm"].values[0]
			im = ax.imshow(cm, cmap="Blues") if m != "fedavg" else ax.imshow(cm, cmap="Greens")
			for (i, j), v in np.ndenumerate(cm):
				ax.text(j, i, int(v), ha='center', va='center', color='black')
			ax.set_title(f"{m} CM")
			ax.set_xlabel("Predicted")
			ax.set_ylabel("Actual")
			ax.set_xticks([0, 1]); ax.set_yticks([0, 1])

		ax_metrics = fig.add_subplot(gs[1, 0:2])
		for metric_name, color in [("accuracy", "#1f77b4"), ("f1", "#ff7f0e"), ("auc", "#2ca02c"), ("log_loss", "#d62728")]:
			if metric_name in res_df.columns:
				vals = res_df.set_index("model")[metric_name].reindex(ordered)
				ax_metrics.plot(range(len(ordered)), vals.values, marker='o', label=metric_name, color=color)
		ax_metrics.set_xticks(range(len(ordered)))
		ax_metrics.set_xticklabels(ordered, rotation=0)
		ax_metrics.set_title("Metrics Comparison (higher better; loss lower)")
		ax_metrics.grid(True, alpha=0.3)
		ax_metrics.legend()

		ax_roc = fig.add_subplot(gs[1, 2:4])
		for m in ordered:
			p = probs.get(m)
			if p is None:
				continue
			try:
				fpr, tpr, _ = roc_curve(y, p)
				auc_val = roc_auc_score(y, p)
				ax_roc.plot(fpr, tpr, label=f"{m} (AUC={auc_val:.3f})")
			except Exception:
				continue
		ax_roc.plot([0, 1], [0, 1], '--', color='gray', alpha=0.6)
		ax_roc.set_xlabel("FPR")
		ax_roc.set_ylabel("TPR")
		ax_roc.set_title("ROC Curves")
		ax_roc.legend()

		fig.suptitle(f"{disease.title()} - Pre vs Post FedAvg (Consolidated)")
		fig.tight_layout(rect=[0, 0.03, 1, 0.95])
		fig_path = FIG_DIR / f"{disease}_summary.png"
		plt.savefig(fig_path, bbox_inches="tight")
		plt.close(fig)
		return res_df.drop(columns=["cm"])  # drop cm from tabular results

	return pd.DataFrame()


all_results = []
for dis in DISEASES.keys():
	res = evaluate_disease(dis)
	if not res.empty:
		all_results.append(res)

if all_results:
	summary = pd.concat(all_results, ignore_index=True)
	csv_out = FIG_DIR / "metrics_summary.csv"
	summary.to_csv(csv_out, index=False)
	print(f"Saved summary to {csv_out}\nFigures saved under {FIG_DIR}")

