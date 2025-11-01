# 🏥 Health Analysis System using Federated Learning and Cloud

A comprehensive health risk prediction system that leverages **Federated Learning** to train models across multiple hospitals while maintaining data privacy. The system uses federated averaging (FedAvg) to aggregate models from different hospitals and provides an intuitive web interface for disease prediction, AI-powered health tips, and appointment management.

## 🎯 Overview

This project demonstrates a production-ready health analysis system that:

- **Trains ML models** using Logistic Regression for 5 diseases (Anemia, Asthma, Breast Cancer, Diabetes, Stroke)
- **Implements Federated Learning** to aggregate models from 3 hospitals without sharing raw patient data
- **Provides web-based interface** for predictions, AI health tips (Gemini API), and appointment scheduling
- **Simulates cloud-based aggregation** locally using FedAvg algorithm
- **Ensures data privacy** by keeping hospital data local and only sharing model weights

### Core Concepts

- **Federated Learning**: Distributed machine learning where models are trained locally at each hospital and only model weights are aggregated
- **FedAvg Algorithm**: Federated Averaging algorithm that combines weights from multiple hospitals using weighted or simple averaging
- **Model Aggregation**: Process of combining multiple hospital models into a single, more robust global model

## ✨ Key Features

### 🔬 Machine Learning
- ✅ Multi-disease prediction models (5 diseases)
- ✅ Federated Learning with 3 hospital participation
- ✅ Weighted and Simple averaging algorithms
- ✅ Logistic Regression models with sklearn compatibility
- ✅ Automatic feature engineering and encoding

### 🌐 Web Application
- ✅ Modern, responsive UI with glassmorphism design
- ✅ User authentication (Login/Signup with role-based access)
- ✅ Admin dashboard for appointment management
- ✅ Real-time health risk predictions
- ✅ AI-powered health tips using Google Gemini API
- ✅ Appointment scheduling system

### 🔒 Security & Privacy
- ✅ Local data storage (hospital data never leaves source)
- ✅ Only model weights are shared (not raw data)
- ✅ User authentication with JSON-based storage
- ✅ Role-based access control (Admin/User)



## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Application Layer                    │
│  (Flask - Welcome, Login, Dashboard, Prediction Forms)      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Application Logic Layer (app.py)               │
│  • User Authentication • Form Processing                    │
│  • Feature Mapping • Prediction Orchestration               │
│  • AI Tips Integration (Gemini API)                         │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│         Federated Learning Simulator Layer                  │
│     (backend/federated_learning/fedavg_simulator.py)        │
│  • Simulates cloud aggregation                              │
│  • FedAvg algorithm implementation                          │
│  • Weighted averaging support                               │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Model Training Layer                           │
│      (backend/model/train_*.py scripts)                     │
│  • Hospital-specific model training                        │
│  • Weight serialization (JSON)                             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Data Layer                                │
│  • Raw datasets (data/raw/)                                │
│  • Hospital-split data (data/hospital/)                    │
│  • Model weights (data/weights/)                           │
│  • User & Contact data (data/*.json)                       │
└─────────────────────────────────────────────────────────────┘
```



## 💻 Technology Stack

### Backend
- **Flask 3.0.2**: Web framework for application server
- **scikit-learn 1.3.2**: Machine learning algorithms (Logistic Regression)
- **pandas 2.1.4**: Data manipulation and analysis
- **numpy 1.26.4**: Numerical computing

### AI & Cloud Services
- **google-generativeai**: Google Gemini API for AI health tips

### Frontend
- **HTML5/CSS3**: Modern responsive design
- **JavaScript**: Dynamic form interactions
- **Google Fonts**: Inter & Poppins typography

### Data Storage
- **JSON**: Model weights, user data, appointments



## 🚀 Installation & Setup

### Prerequisites

- **Python 3.8+** (Recommended: Python 3.10+)
- **pip** (Python package manager)
- **Git** (for cloning repository)

### Step-by-Step Setup

#### 1. Clone the Repository

```powershell
git clone https://github.com/your-username/Health-Analysis-using-Federated-Learning-and-Cloud.git
cd Health-Analysis-using-Federated-Learning-and-Cloud
```

#### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv myenv
.\myenv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv myenv
source myenv/bin/activate
```

#### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

#### 4. Set Up Gemini API Key

**Option A: Environment Variable (Recommended)**
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY = "your-api-key-here"

# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"
```

**Option B: Add to activate.bat** (Windows only)
Edit `myenv\Scripts\activate.bat` and add:
```batch
set GEMINI_API_KEY=your-api-key-here
```

#### 5. Prepare Data

Place your raw CSV files in `data/raw/`:
- `anemia.csv`
- `asthma.csv`
- `breast_cancer.csv`
- `diabetes.csv`
- `stroke.csv`

#### 6. Split Data for Hospitals

```powershell
python backend/constants/split.py
```

This creates hospital-specific datasets in `data/hospital/`.

#### 7. Train Models

Train models for all diseases:
```powershell
python backend/model/train_anemia.py
python backend/model/train_asthma.py
python backend/model/train_breast_cancer.py
python backend/model/train_diabetes.py
python backend/model/train_stroke.py
```

Or train all at once (if you create a script):
```powershell
# Windows
Get-ChildItem backend\model\train_*.py | ForEach-Object { python $_.FullName }
```

#### 8. Run the Application

```powershell
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000**



## 📁 Project Structure

```
Health-Analysis-using-Federated-Learning-and-Cloud/
│
├── app.py                          # Main Flask application
│
├── backend/
│   ├── constants/
│   │   └── split.py               # Data splitting and cleaning utility
│   │
│   ├── federated_learning/
│   │   ├── __init__.py            # Package initializer
│   │   └── fedavg_simulator.py    # FedAvg algorithm simulator
│   │
│   └── model/
│       ├── train_anemia.py        # Anemia model training
│       ├── train_asthma.py        # Asthma model training
│       ├── train_breast_cancer.py # Breast cancer model training
│       ├── train_diabetes.py      # Diabetes model training
│       └── train_stroke.py        # Stroke model training
│
├── data/
│   ├── raw/                       # Raw CSV datasets
│   │   ├── anemia.csv
│   │   ├── asthma.csv
│   │   ├── breast_cancer.csv
│   │   ├── diabetes.csv
│   │   └── stroke.csv
│   │
│   ├── hospital/                  # Hospital-specific datasets
│   │   ├── Hospital A/
│   │   │   ├── anemia.csv
│   │   │   ├── asthma.csv
│   │   │   ├── breast_cancer.csv
│   │   │   ├── diabetes.csv
│   │   │   └── stroke.csv
│   │   ├── Hospital B/
│   │   └── Hospital C/
│   │
│   ├── weights/                   # Trained model weights (JSON)
│   │   ├── anemia/
│   │   │   ├── hospital_a_weights.json
│   │   │   ├── hospital_b_weights.json
│   │   │   └── hospital_c_weights.json
│   │   ├── asthma/
│   │   ├── breast_cancer/
│   │   ├── diabetes/
│   │   └── stroke/
│   │
│   ├── users.json                 # User authentication data
│   └── contacts.json              # Appointment submissions
│
├── static/
│   ├── css/
│   │   └── styles.css            # Global stylesheet
│   └── img/
│       ├── icon.png              # Application icon
│       ├── airobot.png           # AI robot image
│       └── *.jpg                 # Background images
│
├── templates/
│   ├── welcome.html              # Landing page
│   ├── login.html                # User login page
│   ├── signup.html               # User registration
│   ├── user_dashboard.html       # User home dashboard
│   ├── index.html                # Disease selection dashboard
│   ├── form.html                 # Prediction input form
│   ├── result.html               # Prediction results
│   ├── ai_tips.html              # AI health tips chatbot
│   ├── contact.html              # Appointment scheduling
│   └── admin_dashboard.html      # Admin panel
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # MIT License
```



## 📚 Module Documentation

### 🔹 Main Application (`app.py`)

The core Flask application orchestrating all system functionality.

#### **Key Functions**

##### `load_features(disease_folder)`
- **Purpose**: Loads feature names from saved model weights
- **Parameters**: `disease_folder` (str) - Disease identifier
- **Returns**: List of feature names
- **Usage**: Used to dynamically generate form fields

##### `load_all_hospital_weights(disease_folder)`
- **Purpose**: Loads model weights from all participating hospitals
- **Parameters**: `disease_folder` (str) - Disease identifier
- **Returns**: List of weight dictionaries from each hospital
- **Usage**: Used for federated averaging

##### `average_weights(weights_list)`
- **Purpose**: Performs simple averaging of model weights
- **Parameters**: `weights_list` (List[Dict]) - Hospital weights
- **Returns**: Aggregated weight dictionary
- **Algorithm**: Simple arithmetic mean of coefficients and intercepts

##### `map_form_inputs_to_features(disease_folder, form_inputs)`
- **Purpose**: Converts user-friendly form inputs to model-compatible feature vectors
- **Parameters**: 
  - `disease_folder` (str) - Disease identifier
  - `form_inputs` (Dict) - User form submission data
- **Returns**: Feature vector (List[float]) in correct order
- **Key Features**:
  - Handles age bucket encoding for asthma (age → Age_0-9, Age_10-19, etc.)
  - One-hot encodes categorical features (gender, smoking_history)
  - Maps checkboxes to binary features
  - Disease-specific transformations

##### `predict_with_averaged_model(disease_folder, feature_vector, use_fedavg_simulator=True)`
- **Purpose**: Makes predictions using federated averaged weights
- **Parameters**:
  - `disease_folder` (str) - Disease identifier
  - `feature_vector` (List[float]) - Input features
  - `use_fedavg_simulator` (bool) - Use FedAvg simulator or simple average
- **Returns**: Prediction dictionary with:
  - `prediction`: "Positive" or "Negative"
  - `probability`: Probability of positive class (0-1)
  - `confidence`: Confidence in prediction (0-1)
  - `prob_negative`: Probability of negative class
- **Algorithm**: 
  1. Aggregates weights using FedAvg
  2. Computes decision function: `z = w^T * x + b`
  3. Applies sigmoid: `P(y=1|x) = 1 / (1 + exp(-z))`
  4. Classifies with threshold 0.5

##### `get_user_friendly_fields(disease_folder)`
- **Purpose**: Generates user-friendly form field definitions
- **Returns**: List of field dictionaries with type, label, validation
- **Field Types**: `number`, `select`, `checkbox`

#### **Routes**

##### Authentication Routes
- `GET/POST /` - Welcome page
- `GET/POST /login` - User login with role selection
- `GET/POST /signup` - User registration with admin code validation
- `GET /logout` - Session termination

##### User Routes
- `GET /user` - User dashboard (Predict, AI Tips, Contact)
- `GET/POST /ai-tips` - AI health tips chatbot (Gemini integration)
- `GET/POST /contact` - Appointment scheduling form

##### Prediction Routes
- `GET /dashboard` - Disease selection dashboard
- `GET/POST /form/<disease>` - Disease-specific prediction form
- `GET /result/<disease>/<prediction>` - Prediction results display

##### Admin Routes
- `GET /admin` - Admin dashboard (appointment management)
- `POST /admin/update_status` - Update appointment status



### 🔹 Federated Learning Module (`backend/federated_learning/fedavg_simulator.py`)

Simulates cloud-based federated averaging process locally.

#### **FedAvgSimulator Class**

##### `__init__(disease_folder)`
- Initializes simulator with disease-specific configuration
- Sets up paths for hospital weight files

##### `simulate_cloud_aggregation(simulate_delay=True, delay_seconds=0.5)`
- **Purpose**: Simulates the cloud aggregation workflow
- **Process**:
  1. Fetches weights from each hospital (simulates network requests)
  2. Shows progress messages
  3. Aggregates using FedAvg algorithm
  4. Returns global model
- **Features**:
  - Configurable network delay simulation
  - Console progress output
  - Error handling for missing weights

##### `_fedavg_algorithm(hospital_weights)`
- **Purpose**: Implements Federated Averaging algorithm
- **Formula**: `w_global = (1/K) * Σ(w_k)` (equal weights)
- **Returns**: Aggregated model weights
- **Mathematical Foundation**:
  ```
  For each feature i:
    w_global[i] = (w_A[i] + w_B[i] + w_C[i]) / 3
  
  intercept_global = (b_A + b_B + b_C) / 3
  ```

##### `get_weighted_fedavg(sample_counts=None)`
- **Purpose**: Weighted federated averaging by sample counts
- **Formula**: `w_global = Σ(n_k * w_k) / Σ(n_k)`
- **Use Case**: When hospitals have different dataset sizes
- **Parameters**: `sample_counts` (Dict[str, int]) - Hospital sample counts

#### **Key Concepts**

**Federated Averaging (FedAvg)**:
- Standard algorithm for federated learning
- Combines local models without sharing raw data
- Preserves data privacy
- Improves model generalization

**Aggregation Process**:
```
Hospital A: w_A, b_A  ┐
Hospital B: w_B, b_B  ├─→ Cloud Aggregator ─→ w_global, b_global
Hospital C: w_C, b_C  ┘
```



### 🔹 Model Training Modules (`backend/model/train_*.py`)

Each training script follows a similar pattern:

#### **Common Workflow**

1. **Data Loading**: Load hospital-specific CSV from `data/hospital/`
2. **Feature Engineering**: 
   - Separate features and target
   - One-hot encode categorical variables (if needed)
3. **Train/Test Split**: 80/20 split with stratification
4. **Model Training**: Logistic Regression with balanced class weights
5. **Weight Extraction**: Extract coefficients and intercept
6. **Serialization**: Save weights as JSON in `data/weights/`

#### **Disease-Specific Details**

##### `train_anemia.py`
- **Features**: Gender, Hemoglobin, MCH, MCHC, MCV
- **Target**: Result (binary: 0/1)
- **Model**: Logistic Regression (balanced classes)
- **Preprocessing**: Direct numeric features

##### `train_asthma.py`
- **Features**: 
  - Symptoms: Tiredness, Dry-Cough, Difficulty-in-Breathing, etc.
  - Age buckets: Age_0-9, Age_10-19, Age_20-24, Age_25-59, Age_60+
  - Gender: One-hot encoded (Female, Male)
- **Target**: Binary classification
- **Preprocessing**: One-hot encoding for categorical features

##### `train_diabetes.py`
- **Features**:
  - Numeric: age, hypertension, bmi, HbA1c_level, blood_glucose_level
  - Categorical: gender (Female, Male, Other), smoking_history
- **Target**: diabetes (binary)
- **Preprocessing**: `pd.get_dummies()` for categorical columns

##### `train_stroke.py`
- **Features**: 15+ symptom indicators, Age, Stroke Risk (%)
- **Target**: At Risk (Binary)
- **Preprocessing**: Binary symptom encoding

##### `train_breast_cancer.py`
- **Features**: mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness
- **Target**: diagnosis (binary)
- **Preprocessing**: Direct numeric features

#### **Weight JSON Format**

Each trained model saves weights in this format:
```json
{
  "model": "logistic_regression",
  "hospital": "Hospital A",
  "features": ["feature1", "feature2", ...],
  "coef": [[coef1, coef2, ...]],
  "intercept": [intercept_value],
  "classes": [0, 1]
}
```



### 🔹 Data Processing Module (`backend/constants/split.py`)

Handles data cleaning and hospital distribution.

#### **Key Functions**

- **Data Cleaning**: Removes columns with single value or >95% same value
- **Disease-Specific Cleaning**: 
  - Diabetes: Removes "No Info" from smoking_history
  - Others: Standard cleaning procedures
- **Data Splitting**: Randomly splits data into 3 hospital datasets
- **Output**: Creates `data/hospital/Hospital {A|B|C}/` directories with cleaned CSVs



### 🔹 Web Templates

#### **Authentication Pages**

##### `welcome.html`
- Landing page with login/signup options
- Modern split-screen design
- Background image integration

##### `login.html`
- Split layout: welcome message + form
- Role selection dropdown (User/Admin)
- Transparent glassmorphism design

##### `signup.html`
- User registration with admin code validation
- Dynamic admin code field (shows when "Admin" selected)
- Success message with redirect to login

#### **Dashboard Pages**

##### `user_dashboard.html`
- Three main options: Predict, AI Tips, Contact
- Hospital background with glass cards
- Responsive grid layout

##### `index.html` (Disease Selection)
- Disease selection cards in single row
- Compact design for 5 diseases
- Glassmorphism styling

##### `admin_dashboard.html`
- Statistics cards (Total, New, Active, Scheduled)
- Data table with status management
- Modern table design with hover effects

#### **Functionality Pages**

##### `form.html`
- Dynamic form generation based on disease
- Smart field types (number, select, checkbox)
- Transparent background with readable inputs
- User-friendly labels and placeholders

##### `result.html`
- Enhanced result display with icons
- Confidence bar visualization
- Color-coded by prediction (red/green)
- Action buttons for next steps

##### `ai_tips.html`
- Split-screen chatbot interface
- Robot doctor image on left
- Chat interface on right
- Full-height transparent design

##### `contact.html`
- Appointment scheduling form
- All required fields with validation
- Success/error message handling



## 🔬 Federated Learning Implementation

### How It Works

1. **Local Training**: Each hospital trains model on its own data
   ```
   Hospital A: Train → w_A, b_A
   Hospital B: Train → w_B, b_B
   Hospital C: Train → w_C, b_C
   ```

2. **Weight Sharing**: Only model weights are shared (not raw data)
   ```
   Raw Data: Stays at hospital ❌
   Model Weights: Shared ✅
   ```

3. **Cloud Aggregation**: Simulated cloud server aggregates weights
   ```
   Cloud: FedAvg(w_A, w_B, w_C) → w_global
   ```

4. **Global Model**: Aggregated model used for predictions
   ```
   Prediction: predict(x, w_global)
   ```

### FedAvg Algorithm Details

**Simple Averaging**:
```python
w_global[i] = (w_A[i] + w_B[i] + w_C[i]) / 3
b_global = (b_A + b_B + b_C) / 3
```

**Weighted Averaging** (by sample count):
```python
total_samples = n_A + n_B + n_C
w_global[i] = (n_A*w_A[i] + n_B*w_B[i] + n_C*w_C[i]) / total_samples
b_global = (n_A*b_A + n_B*b_B + n_C*b_C) / total_samples
```

### Privacy Benefits

- ✅ **Data Privacy**: Raw patient data never leaves hospital
- ✅ **HIPAA Compliance**: Only aggregated statistics shared
- ✅ **Decentralized**: No central data repository
- ✅ **Scalable**: Easy to add more hospitals



## 🌐 Web Application Guide

### User Workflow

1. **Welcome Page** → Choose Login or Signup
2. **Signup** → Create account (Admin code: `VSHospital`)
3. **Login** → Select role (User/Admin) and enter credentials
4. **User Dashboard** → Three options:
   - **Predict Chances**: Select disease → Fill form → View results
   - **AI Tips**: Chat with AI doctor for health advice
   - **Contact Hospital**: Schedule appointment

### Admin Workflow

1. **Login** as Admin
2. **Admin Dashboard** → View appointment requests
3. **Update Status** → Mark as New/Active/Scheduled/Resolved
4. **Monitor Statistics** → Track request counts

### Prediction Process

1. Select disease from dashboard
2. Form appears with disease-specific fields
3. Fill required information:
   - **Asthma**: Age, gender, symptoms (checkboxes)
   - **Diabetes**: Age, gender, hypertension, BMI, HbA1c, glucose, smoking
   - **Stroke**: Age, symptoms (checkboxes), optional risk %
   - **Anemia**: Gender, blood test values (Hemoglobin, MCH, MCHC, MCV)
4. Submit → Federated averaging occurs → Prediction displayed
5. View results with confidence percentage



## 📊 Usage Examples

### Training All Models

	```powershell
# Train all disease models
python backend/model/train_anemia.py
python backend/model/train_asthma.py
python backend/model/train_breast_cancer.py
python backend/model/train_diabetes.py
	python backend/model/train_stroke.py
	```

### Testing Federated Averaging

```python
from backend.federated_learning.fedavg_simulator import FedAvgSimulator

# Initialize simulator
simulator = FedAvgSimulator("diabetes")

# Simulate cloud aggregation
aggregated_weights = simulator.simulate_cloud_aggregation(simulate_delay=True)

# Use weighted FedAvg
sample_counts = {"hospital_a": 1000, "hospital_b": 800, "hospital_c": 1200}
weighted_weights = simulator.get_weighted_fedavg(sample_counts)
```

### Making Predictions Programmatically

```python
from app import map_form_inputs_to_features, predict_with_averaged_model

# Example: Diabetes prediction
form_inputs = {
    "age": 45,
    "gender": "Male",
    "hypertension": "0",
    "bmi": 28.5,
    "HbA1c_level": 7.2,
    "blood_glucose_level": 180,
    "smoking_history": "never"
}

# Convert to feature vector
feature_vector = map_form_inputs_to_features("diabetes", form_inputs)

# Make prediction
result = predict_with_averaged_model("diabetes", feature_vector)
print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']*100:.1f}%")
```



## 🔌 API Documentation

### Authentication Endpoints

#### `POST /signup`
- **Purpose**: User registration
- **Body**:
  ```json
  {
    "email": "user@example.com",
    "password": "password123",
    "role": "user" | "admin",
    "admin_code": "VSHospital"  // Required if role=admin
  }
  ```
- **Response**: Redirects to login on success

#### `POST /login`
- **Purpose**: User authentication
- **Body**:
  ```json
  {
    "email": "user@example.com",
    "password": "password123",
    "role": "user" | "admin"
  }
  ```
- **Response**: Redirects to user/admin dashboard

### Prediction Endpoints

#### `POST /form/<disease>`
- **Purpose**: Submit prediction form
- **Parameters**: `disease` - Disease name (Anemia, Asthma, etc.)
- **Body**: Form data with disease-specific fields
- **Response**: Redirects to result page with prediction

#### `GET /result/<disease>/<prediction>?confidence=<percent>`
- **Purpose**: Display prediction results
- **Query Parameters**: `confidence` - Confidence percentage

### Admin Endpoints

#### `POST /admin/update_status`
- **Purpose**: Update appointment status
- **Body**:
  ```json
  {
    "id": 1,
    "status": "new" | "active" | "scheduled" | "resolved"
  }
  ```

### AI Tips Endpoint

#### `POST /ai-tips`
- **Purpose**: Get AI health tips
- **Body**:
  ```json
  {
    "message": "How can I improve my heart health?"
  }
  ```
- **Response**: AI-generated health advice



## 🧪 Testing & Validation

### Model Validation

Each training script uses:
- **Train/Test Split**: 80/20 with stratification
- **Class Balancing**: `class_weight='balanced'` for Logistic Regression
- **Random State**: Fixed seed (42) for reproducibility

### Prediction Accuracy

- Models use sklearn-compatible prediction logic
- Sigmoid function matches `predict_proba` output
- Threshold: 0.5 (standard for binary classification)

### Data Validation

- Feature vector length validation
- Missing value handling
- Type conversion and error handling



## 🔧 Troubleshooting

### Common Issues

#### 1. **Import Errors**
```
Solution: Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

#### 2. **Model Weights Not Found**
```
Solution: Train models first
python backend/model/train_*.py
```

#### 3. **Gemini API Errors**
```
Solution: Check API key is set
$env:GEMINI_API_KEY = "your-key"
```

#### 4. **Feature Vector Mismatch**
```
Error: Feature vector length != coefficient length
Solution: Ensure form inputs match expected features
Check map_form_inputs_to_features() function
```

#### 5. **Prediction Always Same**
```
Solution: Check feature vector mapping
Verify all inputs are being converted correctly
Check model weights are loaded properly
```



## 🚀 Future Enhancements

### Planned Features

- [ ] Real-time model retraining
- [ ] Weight encryption for secure sharing
- [ ] Multi-round federated learning
- [ ] Model versioning and rollback
- [ ] Advanced metrics dashboard
- [ ] Email notifications for appointments
- [ ] Mobile-responsive optimizations
- [ ] API rate limiting
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure/GCP)

### Performance Improvements

- [ ] Caching for model weights
- [ ] Async prediction processing
- [ ] Database integration (PostgreSQL)
- [ ] Redis for session management



## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

