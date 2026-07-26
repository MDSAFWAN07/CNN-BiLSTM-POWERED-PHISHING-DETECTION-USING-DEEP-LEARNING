"""
Phishing URL Detection - Data Preprocessing
Author: AI Phishing Detector
Date: 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
from urllib.parse import urlparse
import re

class URLFeatureExtractor:
    """Extract features from URLs for phishing detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_additional_features(self, url):
        """Extract additional URL features"""
        features = {}
        
        try:
            parsed = urlparse(url)
            
            # Domain features
            features['has_https'] = 1 if parsed.scheme == 'https' else 0
            features['domain_length'] = len(parsed.netloc)
            features['path_length'] = len(parsed.path)
            
            # Suspicious patterns
            features['has_ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
            features['has_at'] = 1 if '@' in url else 0
            features['has_double_slash'] = 1 if '//' in parsed.path else 0
            features['has_dash'] = 1 if '-' in parsed.netloc else 0
            
            # URL entropy (complexity measure)
            features['url_entropy'] = self.calculate_entropy(url)
            
            # Digit ratio
            digits = sum(c.isdigit() for c in url)
            features['digit_ratio'] = digits / len(url) if len(url) > 0 else 0
            
            # Special character count
            special_chars = sum(not c.isalnum() for c in url)
            features['special_char_ratio'] = special_chars / len(url) if len(url) > 0 else 0
            
        except:
            # Default values if parsing fails
            features = {
                'has_https': 0, 'domain_length': 0, 'path_length': 0,
                'has_ip': 0, 'has_at': 0, 'has_double_slash': 0,
                'has_dash': 0, 'url_entropy': 0, 'digit_ratio': 0,
                'special_char_ratio': 0
            }
            
        return features
    
    def calculate_entropy(self, text):
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        entropy = 0
        for x in range(256):
            p_x = float(text.count(chr(x))) / len(text)
            if p_x > 0:
                entropy += - p_x * np.log2(p_x)
        return entropy
    
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        print("Starting preprocessing...")
        print(f"Original dataset shape: {df.shape}")
        
        # Create a copy
        processed_df = df.copy()
        
        # Extract additional features from URLs
        print("Extracting additional URL features...")
        url_features = processed_df['url'].apply(self.extract_additional_features)
        url_features_df = pd.DataFrame(url_features.tolist())
        
        # Combine with existing features
        feature_columns = [col for col in processed_df.columns if col not in ['url', 'status']]
        X = pd.concat([processed_df[feature_columns], url_features_df], axis=1)
        
        # Encode target variable
        y = (processed_df['status'] == 'phishing').astype(int)
        
        print(f"Total features: {X.shape[1]}")
        print(f"Feature names: {X.columns.tolist()}")
        
        return X, y, X.columns.tolist()

def main():
    """Main preprocessing pipeline"""
    print("="*60)
    print("PHISHING DETECTION - DATA PREPROCESSING")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_excel('/mnt/user-data/uploads/DATASET_33_56_.xlsx')
    print(f"   ✓ Loaded {len(df)} records")
    print(f"   ✓ Legitimate URLs: {(df['status'] == 'legitimate').sum()}")
    print(f"   ✓ Phishing URLs: {(df['status'] == 'phishing').sum()}")
    
    # Initialize feature extractor
    extractor = URLFeatureExtractor()
    
    # Preprocess
    print("\n2. Preprocessing and feature extraction...")
    X, y, feature_names = extractor.preprocess_data(df)
    
    # Split data
    print("\n3. Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✓ Training set: {len(X_train)} samples")
    print(f"   ✓ Test set: {len(X_test)} samples")
    
    # Scale features
    print("\n4. Scaling features...")
    extractor.scaler.fit(X_train)
    X_train_scaled = extractor.scaler.transform(X_train)
    X_test_scaled = extractor.scaler.transform(X_test)
    print("   ✓ Features scaled using StandardScaler")
    
    # Save preprocessed data
    print("\n5. Saving preprocessed data...")
    np.save('/home/claude/X_train.npy', X_train_scaled)
    np.save('/home/claude/X_test.npy', X_test_scaled)
    np.save('/home/claude/y_train.npy', y_train)
    np.save('/home/claude/y_test.npy', y_test)
    print("   ✓ Saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
    
    # Save scaler and feature names
    joblib.dump(extractor.scaler, '/home/claude/scaler.pkl')
    joblib.dump(feature_names, '/home/claude/feature_names.pkl')
    print("   ✓ Saved: scaler.pkl, feature_names.pkl")
    
    # Save preprocessing metadata
    metadata = {
        'total_samples': len(df),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'num_features': X.shape[1],
        'feature_names': feature_names,
        'class_distribution': {
            'legitimate': int((y == 0).sum()),
            'phishing': int((y == 1).sum())
        }
    }
    
    with open('/home/claude/preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("   ✓ Saved: preprocessing_metadata.json")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nDataset Summary:")
    print(f"  • Total Features: {X.shape[1]}")
    print(f"  • Training Samples: {len(X_train)}")
    print(f"  • Test Samples: {len(X_test)}")
    print(f"  • Class Balance: Legitimate={metadata['class_distribution']['legitimate']}, Phishing={metadata['class_distribution']['phishing']}")

if __name__ == "__main__":
    main()
