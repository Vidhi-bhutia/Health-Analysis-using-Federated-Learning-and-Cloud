# 🏥 Health Analysis System using Federated Learning

A health risk prediction system that leverages **Federated Learning** to train models across multiple hospitals while maintaining data privacy. The system uses federated averaging (FedAvg) to aggregate models from different hospitals and provides a web interface for disease prediction, AI-powered health tips, and appointment management.

## ✨ Features

- **Multi-disease Prediction**: 5 diseases (Anemia, Asthma, Breast Cancer, Diabetes, Stroke)
- **Federated Learning**: Aggregates models from 3 hospitals without sharing raw patient data
- **Web Application**: Modern UI with user authentication, prediction forms, and admin dashboard
- **AI Health Tips**: Google Gemini API integration for health advice chatbot
- **Appointment Management**: Contact form submissions with admin status tracking

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Health-Analysis-using-Federated-Learning-and-Cloud
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myenv
   # Windows
   .\myenv\Scripts\activate
   # Linux/Mac
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Gemini API Key**
   ```bash
   # Windows PowerShell
   $env:GEMINI_API_KEY = "your-api-key-here"
   
   # Linux/Mac
   export GEMINI_API_KEY="your-api-key-here"
   ```

5. **Prepare data and train models**
   ```bash
   # Split data for hospitals
   python backend/constants/split.py
   
   # Train models
   python backend/model/train_anemia.py
   python backend/model/train_asthma.py
   python backend/model/train_breast_cancer.py
   python backend/model/train_diabetes.py
   python backend/model/train_stroke.py
   ```

6. **Run the application**
   ```bash
   python app.py
   ```
   
   Open browser: **http://127.0.0.1:5000**

## 📁 Project Structure

```
├── app.py                          # Main Flask application
├── backend/
│   ├── constants/
│   │   └── split.py               # Data splitting utility
│   ├── federated_learning/
│   │   └── fedavg_simulator.py    # FedAvg simulation
│   └── model/
│       └── train_*.py             # Model training scripts
├── data/
│   ├── raw/                       # Raw CSV datasets
│   ├── hospital/                  # Hospital-split data
│   ├── weights/                   # Trained model weights
│   ├── users.json                 # User credentials
│   └── contacts.json              # Appointment submissions
├── static/
│   └── css/styles.css             # Global styles
├── templates/                      # HTML templates
└── requirements.txt                # Dependencies
```

## 🔑 Authentication

- **Admin Code**: `VSHospital` (required for admin signup)
- **Roles**: User, Admin
- **Storage**: JSON file (`data/users.json`)

## 💻 Usage

### User Workflow

1. **Welcome** → Login/Signup
2. **Signup** → Create account (Admin code: `VSHospital`)
3. **Login** → Select role and enter credentials
4. **Dashboard** → Choose:
   - **Predict Chances**: Select disease → Fill form → View results
   - **AI Tips**: Chat with AI doctor
   - **Contact Hospital**: Schedule appointment

### Admin Workflow

1. **Login** as Admin
2. **Dashboard** → View appointment requests
3. **Update Status** → Mark as New/Active/Scheduled/Resolved

## 🧠 Federated Learning

The system simulates cloud-based federated averaging locally:

1. **Local Training**: Each hospital trains model on its own data
2. **Weight Sharing**: Only model weights are shared (not raw data)
3. **Aggregation**: FedAvg algorithm combines weights from all hospitals
4. **Prediction**: Uses aggregated model for predictions

**FedAvg Formula**:
```
w_global = (w_A + w_B + w_C) / 3
```

## 🛠️ Technology Stack

- **Backend**: Flask 3.0.2
- **ML**: scikit-learn, pandas, numpy
- **AI**: google-generativeai (Gemini API)
- **Frontend**: HTML5/CSS3, JavaScript

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.
