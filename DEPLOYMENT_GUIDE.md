# 🚀 Complete Deployment Guide - PhishGuard

## Step-by-Step Guide to Deploy on Render and Setup Chrome Extension

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [GitHub Setup](#github-setup)
3. [Render Deployment](#render-deployment)
4. [Testing with Postman](#testing-with-postman)
5. [Chrome Extension Setup](#chrome-extension-setup)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- ✅ Python 3.8+ installed
- ✅ Git installed
- ✅ GitHub account
- ✅ Render account (free tier available at [render.com](https://render.com))
- ✅ Google Chrome browser
- ✅ Postman installed (optional)

---

## GitHub Setup

### Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and log in
2. Click the "+" icon → "New repository"
3. Name it: `phishguard-api`
4. Set to **Public** or **Private**
5. Click "Create repository"

### Step 2: Prepare Your Local Project

```bash
# Navigate to your project directory
cd /path/to/phishguard

# Initialize git (if not already done)
git init

# Create .gitignore file
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.npy
.DS_Store
.vscode/
*.log
EOF

# Add all files
git add .

# Commit
git commit -m "Initial commit - PhishGuard API"

# Add remote (replace with your GitHub URL)
git remote add origin https://github.com/YOUR_USERNAME/phishguard-api.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Verify Files on GitHub

Ensure these files are in your repository:
- ✅ `app.py`
- ✅ `requirements.txt`
- ✅ `bilstm_model.pkl`
- ✅ `xgboost_model.pkl`
- ✅ `rf_model.pkl`
- ✅ `scaler.pkl`
- ✅ `feature_names.pkl`
- ✅ `ensemble_config.json`
- ✅ `README.md`

---

## Render Deployment

### Step 1: Create Render Account

1. Go to [render.com](https://render.com)
2. Sign up with GitHub (recommended)
3. Authorize Render to access your GitHub

### Step 2: Create New Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository:
   - Select "phishguard-api"
   - Click "Connect"

### Step 3: Configure Service

Fill in the following details:

**Basic Settings:**
- **Name**: `phishguard-api` (or your choice)
- **Region**: Choose closest to you
- **Branch**: `main`
- **Root Directory**: Leave blank
- **Environment**: `Python 3`
- **Build Command**: 
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command**:
  ```bash
  gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
  ```

**Instance Type:**
- Select **Free** tier (or paid for better performance)

**Environment Variables** (Optional):
- Click "Add Environment Variable"
- Add any custom variables if needed

### Step 4: Deploy

1. Click **"Create Web Service"**
2. Render will start building your application
3. Wait 5-10 minutes for deployment
4. Once complete, you'll see "Live" status

### Step 5: Get Your API URL

Your API will be available at:
```
https://phishguard-api.onrender.com
```

(Replace with your actual service name)

### Step 6: Test the API

Open your browser and visit:
```
https://your-service-name.onrender.com/health
```

You should see:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "1.0"
}
```

---

## Testing with Postman

### Method 1: Using Postman Application

#### Step 1: Install Postman
Download from [postman.com](https://www.postman.com/downloads/)

#### Step 2: Create New Request

1. Open Postman
2. Click "New" → "HTTP Request"

#### Step 3: Test Health Endpoint

```
Method: GET
URL: https://your-service-name.onrender.com/health
```

Click **"Send"**

Expected Response (200 OK):
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "1.0"
}
```

#### Step 4: Test Prediction Endpoint

```
Method: POST
URL: https://your-service-name.onrender.com/predict
Headers:
  Content-Type: application/json
Body (raw JSON):
{
  "url": "http://secure-paypal-verify.tk/login.php"
}
```

Click **"Send"**

Expected Response (200 OK):
```json
{
  "success": true,
  "result": {
    "url": "http://secure-paypal-verify.tk/login.php",
    "prediction": "PHISHING",
    "is_phishing": true,
    "confidence": 0.9234,
    "risk_level": "HIGH",
    ...
  }
}
```

#### Step 5: Test Legitimate URL

```
Body (raw JSON):
{
  "url": "https://www.google.com"
}
```

Expected: `"prediction": "LEGITIMATE"`

#### Step 6: Test Batch Prediction

```
Method: POST
URL: https://your-service-name.onrender.com/predict/batch
Body (raw JSON):
{
  "urls": [
    "https://www.google.com",
    "http://secure-login-verify.tk",
    "https://github.com",
    "http://paypal-verify-account.info"
  ]
}
```

### Method 2: Using cURL (Command Line)

#### Test Health:
```bash
curl https://your-service-name.onrender.com/health
```

#### Test Prediction:
```bash
curl -X POST https://your-service-name.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://secure-paypal-verify.tk/login.php"}'
```

---

## Chrome Extension Setup

### Step 1: Configure API URL

1. Navigate to the extension folder
2. Open `chrome-extension/popup.js`
3. Find this line:
   ```javascript
   const API_URL = 'YOUR_RENDER_API_URL';
   ```
4. Replace with your Render URL:
   ```javascript
   const API_URL = 'https://your-service-name.onrender.com';
   ```
5. Save the file

6. Also update `chrome-extension/background.js`:
   ```javascript
   apiUrl: 'https://your-service-name.onrender.com'
   ```

### Step 2: Add Extension Icons

1. Create icon files or use online generators:
   - [favicon-generator.org](https://www.favicon-generator.org/)
   - [realfavicongenerator.net](https://realfavicongenerator.net/)

2. Generate 4 sizes:
   - 16x16 → `icon16.png`
   - 32x32 → `icon32.png`
   - 48x48 → `icon48.png`
   - 128x128 → `icon128.png`

3. Place them in `chrome-extension/icons/` folder

**Quick Icon Design Tips:**
- Use shield or lock symbol
- Colors: #667eea (blue-purple) or #764ba2 (purple)
- Simple, recognizable design
- Transparent background

### Step 3: Install Extension in Chrome

1. Open Google Chrome
2. Navigate to: `chrome://extensions/`
3. Enable **"Developer mode"** (toggle in top right)
4. Click **"Load unpacked"**
5. Select your `chrome-extension` folder
6. Click **"Select Folder"**

The extension should now appear in your extensions list!

### Step 4: Pin Extension to Toolbar

1. Click the puzzle piece icon in Chrome toolbar
2. Find "PhishGuard - AI Phishing Detector"
3. Click the pin icon to keep it visible

### Step 5: Test Extension

#### Test 1: Legitimate Website
1. Visit `https://www.google.com`
2. Click the PhishGuard icon
3. Should show: "✅ WEBSITE IS SAFE"

#### Test 2: Suspicious Pattern
1. Visit a test phishing URL (be careful!)
2. Click the PhishGuard icon
3. Should show: "🛑 PHISHING DETECTED"

#### Test 3: Check Features
- ✅ View confidence scores
- ✅ See risk level badge
- ✅ Toggle model details
- ✅ Check response time

---

## Troubleshooting

### API Issues

#### Problem: "Application Error" on Render

**Solutions:**
1. Check Render logs:
   - Go to your service dashboard
   - Click "Logs" tab
   - Look for error messages

2. Verify requirements.txt:
   ```bash
   flask==3.0.0
   flask-cors==4.0.0
   gunicorn==21.2.0
   numpy==1.24.3
   scikit-learn==1.3.0
   joblib==1.3.2
   ```

3. Check if all model files are uploaded to GitHub

#### Problem: "Models not loading"

**Solution:**
Ensure these files exist in your repository:
- bilstm_model.pkl
- xgboost_model.pkl
- rf_model.pkl
- scaler.pkl
- feature_names.pkl
- ensemble_config.json

#### Problem: "API taking too long to respond"

**Solution:**
1. Increase timeout in `app.py`:
   ```python
   app.config['TIMEOUT'] = 120
   ```

2. Or in Render start command:
   ```bash
   gunicorn app:app --bind 0.0.0.0:$PORT --timeout 180
   ```

### Chrome Extension Issues

#### Problem: "Unable to connect to API"

**Solutions:**
1. Verify API URL in popup.js is correct
2. Check API is running (visit health endpoint)
3. Check browser console for errors:
   - Right-click extension icon
   - Select "Inspect popup"
   - Check Console tab

#### Problem: "Extension not loading"

**Solutions:**
1. Check manifest.json syntax
2. Ensure all referenced files exist
3. Reload extension:
   - Go to `chrome://extensions/`
   - Click refresh icon on PhishGuard

#### Problem: "CORS errors"

**Solution:**
Verify in `app.py`:
```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # This line must be present
```

### Performance Issues

#### Problem: "Slow API responses"

**Solutions:**
1. Upgrade Render plan (free tier has limitations)
2. Implement caching for frequently checked URLs
3. Optimize model loading (load once at startup)

#### Problem: "High memory usage"

**Solution:**
Models are loaded once at startup. If issues persist:
1. Use lighter model formats
2. Consider model quantization
3. Upgrade Render instance

---

## Success Checklist

After deployment, verify:

- ✅ API is live on Render
- ✅ Health endpoint returns 200 OK
- ✅ Prediction endpoint works
- ✅ Batch prediction works
- ✅ Chrome extension connects to API
- ✅ Extension shows correct results
- ✅ Risk levels display properly
- ✅ Confidence scores are accurate
- ✅ Model details toggle works
- ✅ Notifications work for high-risk sites

---

## Next Steps

1. **Monitor Your API**
   - Check Render dashboard regularly
   - Monitor request logs
   - Track response times

2. **Improve Extension**
   - Add custom settings page
   - Implement URL whitelist/blacklist
   - Add usage statistics

3. **Share Your Work**
   - Publish on GitHub
   - Share on social media
   - Submit to Chrome Web Store (optional)

---

## Support

If you encounter issues:

1. Check Render logs first
2. Review Chrome console errors
3. Test API with Postman
4. Check GitHub Issues
5. Contact support

---

**🎉 Congratulations! Your PhishGuard system is now live!**

