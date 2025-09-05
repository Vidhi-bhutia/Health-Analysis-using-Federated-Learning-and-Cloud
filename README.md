# Health Analysis using Federated Learning and Cloud

This project demonstrates federated learning for health data analysis across multiple hospitals, using Python and scikit-learn. It simulates distributed model training for five diseases: anemia, asthma, breast cancer, diabetes, and stroke.

## Project Structure

```
├── backend/
│   ├── constants/
│   │   └── split.py           # Script to split and clean raw data into hospital datasets
│   └── model/
│       ├── train_anemia.py        # Train anemia model for each hospital
│       ├── train_asthma.py        # Train asthma model for each hospital
│       ├── train_breast_cancer.py # Train breast cancer model for each hospital
│       ├── train_diabetes.py      # Train diabetes model for each hospital
│       └── train_stroke.py        # Train stroke model for each hospital
├── data/
│   ├── raw/                   # Raw datasets for each disease
|   ├── weights/               # Model weights for each hospital and disease
│   └── hospital/              # Hospital-specific datasets (created by split.py)
│       ├── Hospital A/
│       ├── Hospital B/
│       └── Hospital C/
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── LICENSE
```

## Setup Instructions

1. **Clone the repository:**
	 ```powershell
	 git clone https://github.com/Vidhi-bhutia/Health-Analysis-using-Federated-Learning-and-Cloud.git
	 cd Health-Analysis-using-Federated-Learning-and-Cloud
	 ```

2. **Create and activate a virtual environment:**
	 ```powershell
	 python -m venv myenv
	 .\myenv\Scripts\activate
	 ```

3. **Install dependencies:**
	 ```powershell
	 pip install -r requirements.txt
	 ```

## Data Preparation

1. Place your raw CSV files for each disease in the `data/raw/` directory. The expected filenames are:
	 - anemia.csv
	 - asthma.csv
	 - breast_cancer.csv
	 - diabetes.csv
	 - stroke.csv

2. **Split and clean the data for each hospital:**
	 ```powershell
	 python backend/constants/split.py
	 ```
	 This will create `Hospital A/`, `Hospital B/`, and `Hospital C/` folders with disease-specific CSVs.

## Model Training Scripts

Each script in `backend/model/` trains a logistic regression model for a specific disease using each hospital's data. The resulting model weights are saved as JSON files in the `weights/` directory.

### `train_anemia.py`
- Trains a logistic regression model for anemia for each hospital.
- Saves weights to `weights/anemia/` as JSON files.
- **Run:**
	```powershell
	python backend/model/train_anemia.py
	```

### `train_asthma.py`
- Trains a logistic regression model for asthma for each hospital.
- Converts severity columns to a binary target.
- Saves weights to `weights/asthma/` as JSON files.
- **Run:**
	```powershell
	python backend/model/train_asthma.py
	```

### `train_breast_cancer.py`
- Trains a logistic regression model for breast cancer for each hospital.
- Saves weights to `weights/breast_cancer/` as JSON files.
- **Run:**
	```powershell
	python backend/model/train_breast_cancer.py
	```

### `train_diabetes.py`
- Trains a logistic regression model for diabetes for each hospital.
- One-hot encodes categorical features (e.g., gender, smoking_history).
- Saves weights to `weights/diabetes/` as JSON files.
- **Run:**
	```powershell
	python backend/model/train_diabetes.py
	```

### `train_stroke.py`
- Trains a logistic regression model for stroke for each hospital.
- Saves weights to `weights/stroke/` as JSON files.
- **Run:**
	```powershell
	python backend/model/train_stroke.py
	```

## Weights and Outputs

- After running the training scripts, model weights for each hospital and disease are saved as JSON files in the `weights/` directory, e.g.:
	- `weights/anemia/hospital_a_weights.json`
	- `weights/asthma/hospital_b_asthma.json`
	- ...etc.

## Requirements

See `requirements.txt` for all dependencies. Main packages:
- scikit-learn
- pandas
- numpy
- Flask (for future API integration)
- jupyter (optional, for experiments)

## License

See [LICENSE](LICENSE).