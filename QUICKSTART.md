# ⚡ Quick Start Guide - PhishGuard

## Get Started in 10 Minutes

---

## 🎯 Overview

This guide will help you:
1. ✅ Preprocess the dataset
2. ✅ Train the AI models
3. ✅ Run the API locally
4. ✅ Test with Postman
5. ✅ Install Chrome Extension
6. ✅ Deploy to Render

**Total Time: ~10 minutes**

---

## 📦 Step 1: Setup (2 minutes)

```bash
# Clone or navigate to project directory
cd phishguard

# Install dependencies
pip install -r requirements.txt
```

**Files Needed:**
- ✅ DATASET_33_56_.xlsx (provided)
- ✅ All Python scripts
- ✅ requirements.txt

---

## 🔧 Step 2: Preprocess Data (1 minute)

```bash
python preprocess.py
```

**Expected Output:**
```
============================================================
PHISHING DETECTION - DATA PREPROCESSING
============================================================

✓ Loaded 11430 records
✓ Training set: 9144 samples
✓ Test set: 2286 samples
✓ Features scaled using StandardScaler
✓ Saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy

PREPROCESSING COMPLETED SUCCESSFULLY!
```

**Generated Files:**
- X_train.npy
- X_test.npy
- y_train.npy
- y_test.npy
- scaler.pkl
- feature_names.pkl
- preprocessing_metadata.json

---

## 🤖 Step 3: Train Models (3 minutes)

```bash
python train_model_final.py
```

**Expected Output:**
```
================================================================================
PHISHING DETECTION - HYBRID MODEL TRAINING
================================================================================

✓ Neural Network trained successfully!
✓ Gradient Boosting trained successfully!
✓ Random Forest trained successfully!

┌─────────────────────────┬──────────┬───────────┬─────────┐
│ 🏆 ENSEMBLE (Best)      │ 0.9204   │ 0.9103    │ 0.9326  │
└─────────────────────────┴──────────┴───────────┴─────────┘

Ready for Deployment! 🚀
```

**Generated Files:**
- bilstm_model.pkl
- xgboost_model.pkl
- rf_model.pkl
- ensemble_config.json
- model_metadata.json
- training_summary.txt

---

## 🚀 Step 4: Run API Locally (1 minute)

```bash
python app.py
```

**Expected Output:**
```
============================================================
🚀 PHISHING DETECTION API STARTING
============================================================
Server: Flask Development Server
Host: 0.0.0.0
Port: 5000

Endpoints:
  GET  /              - API information
  GET  /health        - Health check
  POST /predict       - Predict single URL
  POST /predict/batch - Predict multiple URLs
============================================================

 * Running on http://0.0.0.0:5000
```

**Test in Browser:**
Open: `http://localhost:5000/health`

---

## 🧪 Step 5: Test with Postman (2 minutes)

### Option A: Using test_api.py

```bash
# In a new terminal (keep API running)
python test_api.py
```

### Option B: Manual Postman Testing

**Test 1: Health Check**
```
GET http://localhost:5000/health
```

**Test 2: Check Legitimate URL**
```
POST http://localhost:5000/predict
Content-Type: application/json

{
  "url": "https://www.google.com"
}
```

**Test 3: Check Suspicious URL**
```
POST http://localhost:5000/predict
Content-Type: application/json

{
  "url": "http://secure-paypal-verify.tk/login.php"
}
```

---

## 🌐 Step 6: Install Chrome Extension (1 minute)

### Quick Setup:

1. **Configure API URL**
   ```bash
   # Edit chrome-extension/popup.js
   # Change line 4:
   const API_URL = 'http://localhost:5000';
   ```

2. **Add Icons** (optional for testing)
   - For now, skip or use placeholder icons

3. **Load Extension**
   - Open Chrome
   - Go to: `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select `chrome-extension` folder

4. **Test Extension**
   - Visit `https://www.google.com`
   - Click PhishGuard icon
   - See "✅ WEBSITE IS SAFE"

---

## ☁️ Step 7: Deploy to Render (10 minutes)

### Quick Deploy:

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "PhishGuard - Ready for deployment"
   git remote add origin https://github.com/YOUR_USERNAME/phishguard-api.git
   git push -u origin main
   ```

2. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - New → Web Service
   - Connect GitHub repo
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - Click "Create Web Service"
   - Wait 5-10 minutes

3. **Get Your API URL**
   ```
   https://your-service-name.onrender.com
   ```

4. **Update Chrome Extension**
   ```javascript
   // Edit chrome-extension/popup.js
   const API_URL = 'https://your-service-name.onrender.com';
   ```

5. **Reload Extension**
   - Go to `chrome://extensions/`
   - Click reload icon on PhishGuard
   - Test with any website!

---

## ✅ Verification Checklist

After completing all steps:

**Local Testing:**
- [ ] Preprocessing completed successfully
- [ ] Models trained with >90% accuracy
- [ ] API runs on localhost:5000
- [ ] Health endpoint returns 200
- [ ] Prediction endpoint works
- [ ] Extension connects to local API

**Deployment:**
- [ ] Code pushed to GitHub
- [ ] Render service is Live
- [ ] API accessible via HTTPS
- [ ] Postman tests pass on deployed API
- [ ] Chrome extension connects to deployed API
- [ ] Extension shows correct predictions

---

## 🎓 Quick Reference

### Most Common Commands

```bash
# Start API
python app.py

# Test API
python test_api.py

# Check API status
curl http://localhost:5000/health

# Test prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.google.com"}'
```

### File Locations

```
phishguard/
├── app.py                    # Main API file
├── preprocess.py            # Data preprocessing
├── train_model_final.py     # Model training
├── requirements.txt         # Dependencies
├── *.pkl                    # Trained models
├── *.json                   # Configuration files
└── chrome-extension/        # Browser extension
```

---

## 🆘 Quick Troubleshooting

### Issue: "Module not found"
```bash
pip install -r requirements.txt
```

### Issue: "API not responding"
```bash
# Check if running
ps aux | grep python

# Restart API
python app.py
```

### Issue: "Extension not working"
1. Check API URL in popup.js
2. Reload extension
3. Check browser console (F12)

### Issue: "Models not loading"
Make sure all .pkl files are in the same directory as app.py

---

## 📞 Need Help?

- 📖 Full Documentation: See README.md
- 🚀 Deployment Guide: See DEPLOYMENT_GUIDE.md
- 🐛 Report Issues: GitHub Issues
- 💬 Contact: your.email@example.com

---

## 🎉 Next Steps

Once everything is working:

1. **Improve Models**
   - Try different hyperparameters
   - Add more features
   - Collect more training data

2. **Enhance Extension**
   - Add custom settings
   - Implement URL whitelist
   - Add usage analytics

3. **Share Your Work**
   - Star the repository
   - Share on social media
   - Contribute improvements

---

**Happy Phishing Detection! 🛡️**

