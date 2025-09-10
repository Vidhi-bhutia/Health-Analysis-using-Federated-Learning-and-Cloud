from flask import Flask, render_template, request, redirect, url_for
import os, json

app = Flask(__name__)

# Available diseases and corresponding folder names
DISEASES = {
    "Anemia": "anemia",
    "Asthma": "asthma",
    "Breast Cancer": "breast_cancer",
    "Diabetes": "diabetes",
    "Stroke": "stroke"
}

def load_features(disease_folder):
    """Load feature names from hospital_a_weights.json for the given disease"""
    weights_path = os.path.join("data", "weights", disease_folder, "hospital_a_weights.json")
    if not os.path.exists(weights_path):
        return []
    
    with open(weights_path, "r") as f:
        data = json.load(f)
    return data.get("features", [])

@app.route("/")
def dashboard():
    return render_template("index.html", diseases=DISEASES.keys())

@app.route("/form/<disease>", methods=["GET", "POST"])
def form(disease):
    disease_folder = DISEASES.get(disease)
    if not disease_folder:
        return "Disease not found", 404
    
    # Load features dynamically
    features = load_features(disease_folder)

    if request.method == "POST":
        inputs = request.form.to_dict()

        # TODO: Real prediction with model weights
        prediction = "Positive" if sum(len(v) for v in inputs.values()) % 2 == 0 else "Negative"

        return redirect(url_for("result", disease=disease, prediction=prediction))
    
    return render_template("form.html", disease=disease, features=features)

@app.route("/result/<disease>/<prediction>")
def result(disease, prediction):
    return render_template("result.html", disease=disease, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
