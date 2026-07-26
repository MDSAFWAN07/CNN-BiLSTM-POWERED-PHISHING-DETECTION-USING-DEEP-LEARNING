"""
API Testing Script - Phishing Detection
Test the Flask API endpoints
"""

import requests
import json

# API Base URL (change this when deployed)
API_URL = "http://localhost:5000"

def print_result(title, response):
    """Pretty print API response"""
    print("\n" + "="*70)
    print(f"TEST: {title}")
    print("="*70)
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    print("="*70)

def test_api():
    """Run API tests"""
    print("\n" + "█"*70)
    print("     PHISHING DETECTION API - TESTING SUITE")
    print("█"*70)
    
    # Test 1: Health Check
    try:
        response = requests.get(f"{API_URL}/health")
        print_result("Health Check", response)
    except Exception as e:
        print(f"✗ Health check failed: {e}")
    
    # Test 2: API Home
    try:
        response = requests.get(f"{API_URL}/")
        print_result("API Home", response)
    except Exception as e:
        print(f"✗ API home failed: {e}")
    
    # Test 3: Predict Legitimate URL
    try:
        test_url = "https://www.google.com"
        response = requests.post(
            f"{API_URL}/predict",
            json={"url": test_url},
            headers={"Content-Type": "application/json"}
        )
        print_result(f"Predict Legitimate URL: {test_url}", response)
    except Exception as e:
        print(f"✗ Legitimate URL prediction failed: {e}")
    
    # Test 4: Predict Suspicious URL
    try:
        test_url = "http://secure-paypal-verify.tk/login.php"
        response = requests.post(
            f"{API_URL}/predict",
            json={"url": test_url},
            headers={"Content-Type": "application/json"}
        )
        print_result(f"Predict Suspicious URL: {test_url}", response)
    except Exception as e:
        print(f"✗ Suspicious URL prediction failed: {e}")
    
    # Test 5: Predict Phishing-like URL
    try:
        test_url = "http://secure-account-verify.com-payment-update.info/verify"
        response = requests.post(
            f"{API_URL}/predict",
            json={"url": test_url},
            headers={"Content-Type": "application/json"}
        )
        print_result(f"Predict Phishing-like URL: {test_url}", response)
    except Exception as e:
        print(f"✗ Phishing URL prediction failed: {e}")
    
    # Test 6: Batch Prediction
    try:
        test_urls = [
            "https://www.amazon.com",
            "http://secure-login-verify.tk",
            "https://github.com",
            "http://paypal-verify-account.info"
        ]
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"urls": test_urls},
            headers={"Content-Type": "application/json"}
        )
        print_result("Batch Prediction", response)
    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
    
    # Test 7: Invalid Request (Missing URL)
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={},
            headers={"Content-Type": "application/json"}
        )
        print_result("Invalid Request (Missing URL)", response)
    except Exception as e:
        print(f"✗ Invalid request test failed: {e}")
    
    print("\n" + "█"*70)
    print("     TESTING COMPLETED")
    print("█"*70 + "\n")

if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure the Flask API is running on http://localhost:5000")
    print("\nTo start the API, run: python app.py")
    
    input("\nPress Enter to continue with testing...")
    
    test_api()
