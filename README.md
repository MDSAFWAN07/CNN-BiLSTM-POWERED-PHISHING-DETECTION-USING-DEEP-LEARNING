# 🛡️ PhishGuard - AI-Powered Phishing Detection System

## Industrial-Grade Phishing Detection using Hybrid Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-success.svg)]()

A state-of-the-art phishing detection system leveraging hybrid AI models (Bi-LSTM + XGBoost + Random Forest) to protect users from malicious websites with **92.04% accuracy**.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Performance](#-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Chrome Extension](#-chrome-extension)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

---

## ✨ Features

### Core Features
- **Hybrid AI Model**: Ensemble of Neural Network, Gradient Boosting, and Random Forest
- **92.04% Accuracy**: High-precision phishing detection
- **Real-time Analysis**: Instant URL scanning and risk assessment
- **RESTful API**: Easy integration with any application
- **Chrome Extension**: Browser-based protection with industrial UI
- **Confidence Scoring**: Multi-level risk assessment (HIGH/MEDIUM/LOW)
- **Batch Processing**: Analyze multiple URLs simultaneously

### Technical Features
- 28 Advanced URL features extraction
- Shannon entropy calculation for URL complexity
- Domain and path analysis
- Special character pattern recognition
- Weighted ensemble predictions
- Production-ready Flask API
- CORS-enabled for cross-origin requests

---

## 🏗️ Architecture

### Model Pipeline

```
URL Input
   ↓
Feature Extraction (28 features)
   ↓
Feature Scaling (StandardScaler)
   ↓
┌─────────────────────────────────┐
│     Hybrid Ensemble Model       │
├─────────────────────────────────┤
│  • Neural Network (40% weight)  │
│  • Gradient Boosting (35%)      │
│  • Random Forest (25%)          │
└─────────────────────────────────┘
   ↓
Weighted Voting
   ↓
Risk Assessment
   ↓
Result (LEGITIMATE/PHISHING)
```

### Feature Set (28 features)

1. **Basic URL Metrics**: length_url, length_hostname, path_length
2. **Character Counts**: dots, hyphens, at signs, slashes, etc.
3. **Security Indicators**: has_https, has_ip, has_double_slash
4. **Complexity Metrics**: url_entropy, digit_ratio, special_char_ratio
5. **Domain Analysis**: domain_length, has_www, has_com
6. **Web Traffic**: Alexa/Tranco ranking (configurable)

---

## 📊 Performance

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Neural Network | 87.31% | 87.64% | 86.88% | 87.26% | 94.62% |
| Gradient Boosting | 91.99% | 91.03% | 93.18% | 92.09% | 97.71% |
| Random Forest | 91.47% | 90.44% | 92.74% | 91.58% | 97.34% |
| **🏆 Ensemble** | **92.04%** | **91.03%** | **93.26%** | **92.13%** | **97.48%** |

### Confusion Matrix (Ensemble Model)

```
                 Predicted
                 Legit  Phish
Actual  Legit     1038    105
        Phish       77   1066
```

### Dataset Statistics
- **Total Samples**: 11,430
- **Training Samples**: 9,144 (80%)
- **Test Samples**: 2,286 (20%)
- **Class Balance**: 50% Legitimate, 50% Phishing
- **Features**: 28 extracted features

---

## 🚀 Installation

### Prerequisites

```bash
- Python 3.8+
- pip
- Git
```

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/phishguard.git
cd phishguard
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python preprocess.py
python train_model_final.py
```

---

## 💻 Usage

### 1. Data Preprocessing

```bash
python preprocess.py
```

**Output Files:**
- `X_train.npy`, `X_test.npy` - Scaled feature arrays
- `y_train.npy`, `y_test.npy` - Target labels
- `scaler.pkl` - Feature scaler
- `feature_names.pkl` - Feature column names
- `preprocessing_metadata.json` - Preprocessing metadata

### 2. Model Training

```bash
python train_model_final.py
```

**Output Files:**
- `bilstm_model.pkl` - Neural Network model
- `xgboost_model.pkl` - Gradient Boosting model
- `rf_model.pkl` - Random Forest model
- `ensemble_config.json` - Ensemble configuration
- `model_metadata.json` - Model metadata
- `training_summary.txt` - Training summary

### 3. Start Flask API

```bash
python app.py
```

API will be available at: `http://localhost:5000`

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "1.0"
}
```

#### 2. Predict Single URL
```http
POST /predict
Content-Type: application/json

{
  "url": "http://example.com"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "url": "http://example.com",
    "prediction": "LEGITIMATE",
    "is_phishing": false,
    "confidence": 0.9234,
    "risk_level": "LOW",
    "probability_scores": {
      "phishing": 0.0766,
      "legitimate": 0.9234
    },
    "model_scores": {
      "neural_network": 0.0823,
      "gradient_boosting": 0.0734,
      "random_forest": 0.0745
    }
  }
}
```

#### 3. Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
  "urls": [
    "http://example1.com",
    "http://example2.com"
  ]
}
```

**Response:**
```json
{
  "success": true,
  "count": 2,
  "results": [...]
}
```

---

## 🌐 Chrome Extension

### Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (top right)
3. Click "Load unpacked"
4. Select the `chrome-extension` folder
5. The PhishGuard icon will appear in your toolbar

### Configuration

1. Click the PhishGuard icon
2. Go to Settings
3. Enter your API URL (e.g., `https://your-api.onrender.com`)
4. Save settings

### Usage

1. Visit any website
2. Click the PhishGuard icon
3. View instant security analysis
4. See confidence scores and risk levels

### Features

- ✅ Real-time URL scanning
- 🎨 Industrial-grade UI design
- 📊 Detailed model score breakdown
- 🚨 High-risk phishing alerts
- 🔍 Right-click link checking
- ⚙️ Customizable settings

---

## 🌍 Deployment

### Deploy to Render

#### Step 1: Prepare for Deployment

Create `render.yaml`:

```yaml
services:
  - type: web
    name: phishing-detector-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

#### Step 2: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/phishguard.git
git push -u origin main
```

#### Step 3: Deploy on Render

1. Go to [render.com](https://render.com)
2. Create New > Web Service
3. Connect your GitHub repository
4. Select the branch (main)
5. Render will auto-detect Python
6. Click "Create Web Service"
7. Wait for deployment (5-10 minutes)
8. Your API will be available at: `https://your-service.onrender.com`

#### Step 4: Update Chrome Extension

Edit `chrome-extension/popup.js`:

```javascript
const API_URL = 'https://your-service.onrender.com';
```

---

## 🧪 Testing

### Test API Locally

```bash
# Start the API
python app.py

# In another terminal, run tests
python test_api.py
```

### Test with Postman

Import this cURL command:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://secure-paypal-verify.tk/login.php"}'
```

### Test Chrome Extension

1. Load extension in Chrome
2. Visit test websites:
   - Legitimate: `https://www.google.com`
   - Suspicious: `http://secure-login-verify.tk`
3. Click extension icon to see results

---

## 📁 Project Structure

```
phishguard/
│
├── preprocess.py              # Data preprocessing script
├── train_model_final.py       # Model training script
├── app.py                     # Flask API application
├── test_api.py               # API testing script
├── requirements.txt          # Python dependencies
│
├── models/                   # Trained model files
│   ├── bilstm_model.pkl
│   ├── xgboost_model.pkl
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── ensemble_config.json
│   └── model_metadata.json
│
├── chrome-extension/         # Chrome Extension
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   ├── styles.css
│   ├── background.js
│   ├── content.js
│   └── icons/
│       ├── icon16.png
│       ├── icon32.png
│       ├── icon48.png
│       └── icon128.png
│
├── data/                     # Dataset and processed data
│   ├── DATASET_33_56_.xlsx
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
│
└── docs/                     # Documentation
    ├── README.md
    ├── API_DOCS.md
    └── DEPLOYMENT_GUIDE.md
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Authors

- **Your Name** - Initial work - [GitHub](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- Dataset: Phishing Website Dataset from Kaggle/UCI
- Inspiration: Growing cybersecurity threats
- Libraries: scikit-learn, Flask, TensorFlow

---

## 📧 Contact

For questions or support:
- Email: your.email@example.com
- GitHub Issues: [Create an issue](https://github.com/yourusername/phishguard/issues)

---

**⭐ Star this repository if you find it helpful!**

