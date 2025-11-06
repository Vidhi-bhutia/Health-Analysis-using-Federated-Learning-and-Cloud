from flask import Flask, render_template, request, redirect, url_for
from flask import session
import os, json
import google.generativeai as genai
try:
	from backend.federated_learning.fedavg_simulator import FedAvgSimulator
except ImportError:
	# Fallback if module not found
	FedAvgSimulator = None
	print("Warning: FedAvg simulator not available, using simple averaging")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# Available diseases and corresponding folder names
DISEASES = {
	"Anemia": "anemia",
	"Asthma": "asthma",
	"Breast Cancer": "breast_cancer",
	"Diabetes": "diabetes",
	"Stroke": "stroke"
}

# Optional default thresholds (can be tuned per disease)
DEFAULT_THRESHOLDS = {
	"anemia": 0.5,
	"asthma": 0.5,
	"breast_cancer": 0.5,
	"diabetes": 0.5,
	"stroke": 0.5,
}


def load_threshold(disease_folder: str) -> float:
	"""Load per-disease threshold from data/weights/<disease>/threshold.json or use default."""
	thr_path = os.path.join("data", "weights", disease_folder, "threshold.json")
	if os.path.exists(thr_path):
		try:
			with open(thr_path, "r") as f:
				data = json.load(f)
				val = float(data.get("threshold", DEFAULT_THRESHOLDS.get(disease_folder, 0.5)))
				return max(0.0, min(1.0, val))
		except Exception:
			pass
	return DEFAULT_THRESHOLDS.get(disease_folder, 0.5)


def load_features(disease_folder):
	"""Load feature names from hospital_a_weights.json for the given disease"""
	weights_path = os.path.join("data", "weights", disease_folder, "hospital_a_weights.json")
	if not os.path.exists(weights_path):
		return []
	with open(weights_path, "r") as f:
		data = json.load(f)
	return data.get("features", [])


def load_all_hospital_weights(disease_folder):
	"""Load weights from all hospitals for federated averaging"""
	hospitals = ["hospital_a", "hospital_b", "hospital_c"]
	weights_list = []
	for hospital in hospitals:
		weights_path = os.path.join("data", "weights", disease_folder, f"{hospital}_weights.json")
		if os.path.exists(weights_path):
			with open(weights_path, "r") as f:
				weights_list.append(json.load(f))
	return weights_list


def _extract_coef_intercept(weight_obj):
	"""Return (coef_vector, intercept_value) regardless of flat/nested JSON formats."""
	raw_coef = weight_obj.get("coef", [])
	if isinstance(raw_coef, list) and raw_coef and isinstance(raw_coef[0], list):
		coef = raw_coef[0]
	else:
		coef = raw_coef
	raw_intercept = weight_obj.get("intercept", 0.0)
	if isinstance(raw_intercept, list):
		intercept = raw_intercept[0] if raw_intercept else 0.0
	else:
		intercept = raw_intercept
	return coef, float(intercept)


def average_weights(weights_list):
	"""Perform federated averaging of weights; if available, weight by num_samples."""
	if not weights_list:
		return None
	features = weights_list[0]["features"]
	num_features = len(features)
	coef_sum = [0.0] * num_features
	intercept_sum = 0.0
	total_weight = 0.0
	use_weighted = any("num_samples" in w for w in weights_list)
	for w in weights_list:
		coef, intercept = _extract_coef_intercept(w)
		if len(coef) != num_features:
			continue
		weight = float(w.get("num_samples", 1.0)) if use_weighted else 1.0
		for i in range(num_features):
			coef_sum[i] += weight * float(coef[i])
		intercept_sum += weight * float(intercept)
		total_weight += weight
	if total_weight == 0:
		return None
	avg_coef = [c / total_weight for c in coef_sum]
	avg_intercept = intercept_sum / total_weight
	return {
		"features": features,
		"coef": [avg_coef],
		"intercept": [avg_intercept],
		"classes": weights_list[0].get("classes", [0, 1]),
	}


def map_form_inputs_to_features(disease_folder, form_inputs):
	"""Convert user-friendly form inputs to model feature format"""
	features = load_features(disease_folder)
	if not features:
		return {}
	feature_dict = {f: 0.0 for f in features}

	# Disease-specific mappings
	disease_name = disease_folder.lower()

	if disease_name == "asthma":
		age = int(form_inputs.get("age", 0))
		if 0 <= age <= 9:
			feature_dict["Age_0-9"] = 1
		elif 10 <= age <= 19:
			feature_dict["Age_10-19"] = 1
		elif 20 <= age <= 24:
			feature_dict["Age_20-24"] = 1
		elif 25 <= age <= 59:
			feature_dict["Age_25-59"] = 1
		else:
			feature_dict["Age_60+"] = 1
		gender = form_inputs.get("gender", "").lower()
		if gender in ["female", "f"]:
			feature_dict["Gender_Female"] = 1
		elif gender in ["male", "m"]:
			feature_dict["Gender_Male"] = 1
		symptom_map = {
			"tiredness": "Tiredness",
			"dry_cough": "Dry-Cough",
			"difficulty_in_breathing": "Difficulty-in-Breathing",
			"sore_throat": "Sore-Throat",
			"pains": "Pains",
			"nasal_congestion": "Nasal-Congestion",
			"runny_nose": "Runny-Nose",
		}
		for key, feat in symptom_map.items():
			if form_inputs.get(key) in ["1", "yes", "true", "on"]:
				feature_dict[feat] = 1
		if form_inputs.get("none_sympton") in ["1", "yes", "true", "on"]:
			feature_dict["None_Sympton"] = 1
		if form_inputs.get("none_experiencing") in ["1", "yes", "true", "on"]:
			feature_dict["None_Experiencing"] = 1

	elif disease_name == "diabetes":
		age = float(form_inputs.get("age", 0))
		feature_dict["age"] = age
		feature_dict["hypertension"] = float(form_inputs.get("hypertension", 0))
		feature_dict["bmi"] = float(form_inputs.get("bmi", 0))
		feature_dict["HbA1c_level"] = float(form_inputs.get("HbA1c_level", 0))
		feature_dict["blood_glucose_level"] = float(form_inputs.get("blood_glucose_level", 0))
		gender = form_inputs.get("gender", "").lower()
		if gender in ["female", "f"]:
			feature_dict["gender_Female"] = 1
		elif gender in ["male", "m"]:
			feature_dict["gender_Male"] = 1
		elif gender in ["other", "o"]:
			feature_dict["gender_Other"] = 1
		smoking = form_inputs.get("smoking_history", "").lower().replace(" ", "")
		if "current" in smoking:
			feature_dict["smoking_history_current"] = 1
		elif "ever" in smoking:
			feature_dict["smoking_history_ever"] = 1
		elif "former" in smoking:
			feature_dict["smoking_history_former"] = 1
		elif "never" in smoking:
			feature_dict["smoking_history_never"] = 1
		elif "notcurrent" in smoking:
			feature_dict["smoking_history_not current"] = 1

	elif disease_name == "stroke":
		try:
			age = float(form_inputs.get("age", 0))
			feature_dict["Age"] = age
		except (ValueError, TypeError):
			feature_dict["Age"] = 0
		symptom_map = {
			"chest_pain": "Chest Pain",
			"shortness_of_breath": "Shortness of Breath",
			"irregular_heartbeat": "Irregular Heartbeat",
			"fatigue__weakness": "Fatigue & Weakness",
			"dizziness": "Dizziness",
			"swelling_edema": "Swelling (Edema)",
			"pain_in_neck_jaw_shoulder_back": "Pain in Neck/Jaw/Shoulder/Back",
			"excessive_sweating": "Excessive Sweating",
			"persistent_cough": "Persistent Cough",
			"nausea_vomiting": "Nausea/Vomiting",
			"high_blood_pressure": "High Blood Pressure",
			"chest_discomfort_activity": "Chest Discomfort (Activity)",
			"cold_hands_feet": "Cold Hands/Feet",
			"snoring_sleep_apnea": "Snoring/Sleep Apnea",
			"anxiety_feeling_of_doom": "Anxiety/Feeling of Doom",
		}
		for key, feat in symptom_map.items():
			if form_inputs.get(key) in ["1", "yes", "true", "on"]:
				feature_dict[feat] = 1
		try:
			if form_inputs.get("stroke_risk"):
				feature_dict["Stroke Risk (%)"] = float(form_inputs.get("stroke_risk"))
		except (ValueError, TypeError):
			pass

	elif disease_name == "anemia":
		gender = form_inputs.get("gender", "").lower()
		feature_dict["Gender"] = 1 if gender in ["female", "f"] else 0
		feature_dict["Hemoglobin"] = float(form_inputs.get("Hemoglobin", 0))
		feature_dict["MCH"] = float(form_inputs.get("MCH", 0))
		feature_dict["MCHC"] = float(form_inputs.get("MCHC", 0))
		feature_dict["MCV"] = float(form_inputs.get("MCV", 0))

	# Create feature vector in correct order
	feature_vector = [feature_dict.get(f, 0.0) for f in features]
	return feature_vector


def predict_with_averaged_model(disease_folder, feature_vector, use_fedavg_simulator: bool = True):
	"""
	Make prediction using federated averaged weights - matches sklearn-style logistic prediction.
	"""
	if use_fedavg_simulator and FedAvgSimulator is not None:
		try:
			simulator = FedAvgSimulator(disease_folder)
			avg_weights = simulator.simulate_cloud_aggregation(simulate_delay=False)
		except Exception as e:
			print(f"FedAvg simulator error: {e}, falling back to simple averaging")
			weights_list = load_all_hospital_weights(disease_folder)
			if not weights_list:
				return None
			avg_weights = average_weights(weights_list)
	else:
		weights_list = load_all_hospital_weights(disease_folder)
		if not weights_list:
			return None
		avg_weights = average_weights(weights_list)

	if not avg_weights:
		return None

	# Verify feature vector length matches weights
	coef = avg_weights["coef"][0]
	if len(feature_vector) != len(coef):
		print(f"ERROR: Feature vector length {len(feature_vector)} != coefficient length {len(coef)}")
		return None

	intercept = avg_weights["intercept"][0]
	# Logistic prediction
	z = sum(c * f for c, f in zip(coef, feature_vector)) + intercept
	import math
	try:
		if z > 700:
			probability_positive = 1.0
		elif z < -700:
			probability_positive = 0.0
		else:
			probability_positive = 1 / (1 + math.exp(-z))
	except Exception:
		probability_positive = 0.5
	probability_negative = 1.0 - probability_positive

	# Use disease-specific threshold
	threshold = load_threshold(disease_folder)
	if probability_positive >= threshold:
		predicted_label = "Positive"
		confidence_prob = probability_positive
	else:
		predicted_label = "Negative"
		confidence_prob = probability_negative
	return {
		"prediction": predicted_label,
		"probability": probability_positive,
		"confidence": confidence_prob,
		"prob_negative": probability_negative,
		"fedavg_used": use_fedavg_simulator,
	}

USERS_JSON_PATH = os.path.join("data", "users.json")
CONTACTS_JSON_PATH = os.path.join("data", "contacts.json")


def load_users():
	if not os.path.exists(USERS_JSON_PATH):
		os.makedirs(os.path.dirname(USERS_JSON_PATH), exist_ok=True)
		with open(USERS_JSON_PATH, "w") as f:
			json.dump({"users": []}, f)
		return []
	with open(USERS_JSON_PATH, "r") as f:
		try:
			data = json.load(f)
		except json.JSONDecodeError:
			data = {"users": []}
	return data.get("users", [])


def save_users(users):
	os.makedirs(os.path.dirname(USERS_JSON_PATH), exist_ok=True)
	with open(USERS_JSON_PATH, "w") as f:
		json.dump({"users": users}, f, indent=2)


def load_contacts():
	if not os.path.exists(CONTACTS_JSON_PATH):
		os.makedirs(os.path.dirname(CONTACTS_JSON_PATH), exist_ok=True)
		with open(CONTACTS_JSON_PATH, "w") as f:
			json.dump({"submissions": []}, f)
		return []
	with open(CONTACTS_JSON_PATH, "r") as f:
		try:
			data = json.load(f)
		except json.JSONDecodeError:
			data = {"submissions": []}
	return data.get("submissions", [])


def save_contacts(submissions):
	os.makedirs(os.path.dirname(CONTACTS_JSON_PATH), exist_ok=True)
	with open(CONTACTS_JSON_PATH, "w") as f:
		json.dump({"submissions": submissions}, f, indent=2)


@app.route("/")
def welcome():
	return render_template("welcome.html")


@app.route("/dashboard")
def dashboard():
	return render_template("index.html", diseases=DISEASES.keys())


@app.route("/login", methods=["GET", "POST"])
def login():
	message = None
	if request.method == "POST":
		email = request.form.get("email", "").strip().lower()
		password = request.form.get("password", "")
		role = request.form.get("role", "user")

		users = load_users()
		match = next((u for u in users if u.get("email") == email and u.get("password") == password and u.get("role") == role), None)
		if match:
			session["user"] = {"email": email, "role": role}
			if role == "admin":
				return redirect(url_for("admin_home"))
			else:
				return redirect(url_for("user_home"))
		else:
			message = "Invalid credentials. Please try again."

	return render_template("login.html", message=message)


@app.route("/signup", methods=["GET", "POST"])
def signup():
	message = None
	success = False
	if request.method == "POST":
		email = request.form.get("email", "").strip().lower()
		password = request.form.get("password", "")
		role = request.form.get("role", "user")
		admin_code = request.form.get("admin_code", "")

		if not email or not password:
			message = "Email and password are required."
		elif role == "admin" and admin_code != "VSHospital":
			message = "Invalid admin code."
		else:
			users = load_users()
			if any(u.get("email") == email for u in users):
				message = "Email already registered. Please login."
			else:
				users.append({"email": email, "password": password, "role": role})
				save_users(users)
				success = True
				message = "Signup successful. You can now login."

	return render_template("signup.html", message=message, success=success)


@app.route("/logout")
def logout():
	session.pop("user", None)
	return redirect(url_for("welcome"))


def require_role(role):
	user = session.get("user")
	return user and user.get("role") == role


@app.route("/user")
def user_home():
	if not require_role("user") and not require_role("admin"):
		return redirect(url_for("login"))
	return render_template("user_dashboard.html")


@app.route("/admin")
def admin_home():
	if not require_role("admin"):
		return redirect(url_for("login"))
	submissions = [s for s in load_contacts() if s.get("status") != "resolved"]
	return render_template("admin_dashboard.html", submissions=submissions)


@app.route("/admin/update_status", methods=["POST"])
def admin_update_status():
	if not require_role("admin"):
		return redirect(url_for("login"))
	submission_id = request.form.get("id")
	new_status = request.form.get("status")
	submissions = load_contacts()
	for s in submissions:
		if str(s.get("id")) == str(submission_id):
			s["status"] = new_status
			break
	save_contacts(submissions)
	return redirect(url_for("admin_home"))


@app.route("/contact", methods=["GET", "POST"])
def contact():
	if not (require_role("user") or require_role("admin")):
		return redirect(url_for("login"))
	message = None
	success = False
	if request.method == "POST":
		name = request.form.get("name", "").strip()
		age = request.form.get("age", "").strip()
		gender = request.form.get("gender", "").strip()
		phone = request.form.get("phone", "").strip()
		problem = request.form.get("problem", "").strip()
		details = request.form.get("details", "").strip()

		if not name or not age or not gender or not phone or not problem:
			message = "Please fill all required fields."
		else:
			submissions = load_contacts()
			next_id = (max([s.get("id", 0) for s in submissions]) + 1) if submissions else 1
			submissions.append({
				"id": next_id,
				"name": name,
				"age": age,
				"gender": gender,
				"phone": phone,
				"problem": problem,
				"details": details,
				"status": "new"
			})
			save_contacts(submissions)
			success = True
			message = "Submitted successfully. We'll contact you soon."

	return render_template("contact.html", message=message, success=success)


@app.route("/ai-tips", methods=["GET", "POST"])
def ai_tips():
	if not (require_role("user") or require_role("admin")):
		return redirect(url_for("login"))
	
	tips = None
	user_message = None
	error_message = None
	
	if request.method == "POST":
		user_message = request.form.get("message", "").strip()
		if user_message:
			try:
				# Configure Gemini API
				api_key = os.environ.get("GEMINI_API_KEY")
				# Fallback API key if environment variable fails
				if not api_key:
					api_key = "YOUR-API-KEY-HERE"
				if not api_key:
					error_message = "API key not configured. Please contact administrator."
				else:
					genai.configure(api_key=api_key)
					model = genai.GenerativeModel('gemini-2.0-flash')
					
					# Doctor persona prompt
					doctor_prompt = """You are Dr. Sarah Chen, an experienced medical professional with over 15 years of clinical experience, specializing in preventive medicine and health optimization. You hold an MD from Johns Hopkins University and have completed fellowships in Internal Medicine and Preventive Cardiology.

Your approach is:
- Evidence-based and practical
- Empathetic and supportive
- Focused on prevention and lifestyle modifications
- Clear and easy to understand
- Always emphasizing the importance of consulting with healthcare providers for serious concerns

Please provide helpful health tips and guidance based on the user's question. Keep responses concise (1 paragraph would be enough) and always remind users to consult their healthcare provider for medical concerns. Please don't put '*' in the response and if possible break the lines after bullets to display a clean look of the answer. Don't give answer word in 'bold' formatting. 

User question: """
					
					# Generate response
					response = model.generate_content(doctor_prompt + user_message)
					tips = response.text
					
			except Exception as e:
				error_message = f"Sorry, I'm having trouble connecting right now. Please try again later. Error: {str(e)}"
	
	return render_template("ai_tips.html", tips=tips, user_message=user_message, error_message=error_message)


def get_user_friendly_fields(disease_folder):
	"""Generate user-friendly form fields based on disease type"""
	disease_name = disease_folder.lower()
	
	if disease_name == "asthma":
		return [
			{"name": "age", "type": "number", "label": "Age", "required": True, "placeholder": "Enter your age"},
			{"name": "gender", "type": "select", "label": "Gender", "required": True, 
			 "options": [("", "Select Gender"), ("Male", "Male"), ("Female", "Female")]},
			{"name": "tiredness", "type": "checkbox", "label": "Tiredness"},
			{"name": "dry_cough", "type": "checkbox", "label": "Dry Cough"},
			{"name": "difficulty_in_breathing", "type": "checkbox", "label": "Difficulty in Breathing"},
			{"name": "sore_throat", "type": "checkbox", "label": "Sore Throat"},
			{"name": "pains", "type": "checkbox", "label": "Pains"},
			{"name": "nasal_congestion", "type": "checkbox", "label": "Nasal Congestion"},
			{"name": "runny_nose", "type": "checkbox", "label": "Runny Nose"},
		]
	elif disease_name == "diabetes":
		return [
			{"name": "age", "type": "number", "label": "Age", "required": True, "placeholder": "Enter age"},
			{"name": "gender", "type": "select", "label": "Gender", "required": True,
			 "options": [("", "Select Gender"), ("Male", "Male"), ("Female", "Female"), ("Other", "Other")]},
			{"name": "hypertension", "type": "select", "label": "Hypertension", "required": True,
			 "options": [("", "Select"), ("0", "No"), ("1", "Yes")]},
			{"name": "bmi", "type": "number", "label": "BMI", "required": True, "placeholder": "Enter BMI", "step": "0.1"},
			{"name": "HbA1c_level", "type": "number", "label": "HbA1c Level", "required": True, "placeholder": "Enter HbA1c", "step": "0.1"},
			{"name": "blood_glucose_level", "type": "number", "label": "Blood Glucose Level", "required": True, "placeholder": "Enter glucose level"},
			{"name": "smoking_history", "type": "select", "label": "Smoking History", "required": True,
			 "options": [("", "Select"), ("never", "Never"), ("current", "Current"), ("former", "Former"), ("ever", "Ever"), ("not current", "Not Current")]},
		]
	elif disease_name == "stroke":
		return [
			{"name": "age", "type": "number", "label": "Age", "required": True, "placeholder": "Enter age"},
			{"name": "chest_pain", "type": "checkbox", "label": "Chest Pain"},
			{"name": "shortness_of_breath", "type": "checkbox", "label": "Shortness of Breath"},
			{"name": "irregular_heartbeat", "type": "checkbox", "label": "Irregular Heartbeat"},
			{"name": "fatigue__weakness", "type": "checkbox", "label": "Fatigue & Weakness"},
			{"name": "dizziness", "type": "checkbox", "label": "Dizziness"},
			{"name": "swelling_edema", "type": "checkbox", "label": "Swelling (Edema)"},
			{"name": "pain_in_neck_jaw_shoulder_back", "type": "checkbox", "label": "Pain in Neck/Jaw/Shoulder/Back"},
			{"name": "excessive_sweating", "type": "checkbox", "label": "Excessive Sweating"},
			{"name": "persistent_cough", "type": "checkbox", "label": "Persistent Cough"},
			{"name": "nausea_vomiting", "type": "checkbox", "label": "Nausea/Vomiting"},
			{"name": "high_blood_pressure", "type": "checkbox", "label": "High Blood Pressure"},
			{"name": "chest_discomfort_activity", "type": "checkbox", "label": "Chest Discomfort (Activity)"},
			{"name": "cold_hands_feet", "type": "checkbox", "label": "Cold Hands/Feet"},
			{"name": "snoring_sleep_apnea", "type": "checkbox", "label": "Snoring/Sleep Apnea"},
			{"name": "anxiety_feeling_of_doom", "type": "checkbox", "label": "Anxiety/Feeling of Doom"},
			{"name": "stroke_risk", "type": "number", "label": "Stroke Risk (%)", "placeholder": "Optional", "step": "0.1"},
		]
	elif disease_name == "anemia":
		return [
			{"name": "gender", "type": "select", "label": "Gender", "required": True,
			 "options": [("", "Select Gender"), ("Male", "Male"), ("Female", "Female")]},
			{"name": "Hemoglobin", "type": "number", "label": "Hemoglobin (g/dL)", "required": True, "placeholder": "Enter Hemoglobin", "step": "0.1"},
			{"name": "MCH", "type": "number", "label": "MCH (pg)", "required": True, "placeholder": "Enter MCH", "step": "0.1"},
			{"name": "MCHC", "type": "number", "label": "MCHC (g/dL)", "required": True, "placeholder": "Enter MCHC", "step": "0.1"},
			{"name": "MCV", "type": "number", "label": "MCV (fL)", "required": True, "placeholder": "Enter MCV", "step": "0.1"},
		]
	elif disease_name == "breast_cancer":
		# Check what features breast cancer uses
		features = load_features(disease_folder)
		return [{"name": f, "type": "number", "label": f.replace("_", " ").title(), "required": True} for f in features]
	
	# Fallback: return raw features
	features = load_features(disease_folder)
	return [{"name": f, "type": "number", "label": f.replace("_", " ").title(), "required": True} for f in features]


@app.route("/form/<disease>", methods=["GET", "POST"])
def form(disease):
	disease_folder = DISEASES.get(disease)
	if not disease_folder:
		return "Disease not found", 404
	
	if request.method == "POST":
		inputs = request.form.to_dict()
		
		# Convert form inputs to feature vector
		try:
			feature_vector = map_form_inputs_to_features(disease_folder, inputs)
			
			if not feature_vector:
				return redirect(url_for("result", disease=disease, prediction="Error: Invalid input", confidence="0.0%"))
			
			# Make prediction using averaged model
			result = predict_with_averaged_model(disease_folder, feature_vector)
			
			if result:
				prediction = result["prediction"]
				# Confidence as percentage of predicted class probability
				confidence = f"{result['confidence']*100:.1f}%"
				return redirect(url_for("result", disease=disease, prediction=prediction, confidence=confidence))
			else:
				return redirect(url_for("result", disease=disease, prediction="Error: Prediction failed", confidence="0.0%"))
		except Exception as e:
			print(f"ERROR in prediction: {str(e)}")
			import traceback
			traceback.print_exc()
			return redirect(url_for("result", disease=disease, prediction=f"Error: {str(e)}", confidence="0.0%"))
	
	# Get user-friendly fields for form
	form_fields = get_user_friendly_fields(disease_folder)
	return render_template("form.html", disease=disease, form_fields=form_fields)


@app.route("/result/<disease>/<prediction>")
def result(disease, prediction):
	confidence = request.args.get("confidence", "N/A")
	return render_template("result.html", disease=disease, prediction=prediction, confidence=confidence)


if __name__ == "__main__":
	app.run(debug=True)
