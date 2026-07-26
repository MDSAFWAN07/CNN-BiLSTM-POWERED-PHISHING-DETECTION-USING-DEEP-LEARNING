"""
Phishing Detection - Data Preprocessing and Feature Extraction
Author: AI Phishing Detection System
Description: Complete preprocessing pipeline for phishing URL detection
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import json

class URLFeatureExtractor:
    """Extract comprehensive features from URLs for phishing detection"""
    
    def __init__(self):
        self.suspicious_words = [
            'login', 'verify', 'account', 'secure', 'banking', 'update',
            'confirm', 'signin', 'ebayisapi', 'webscr', 'paypal'
        ]
        
    def extract_features(self, url):
        """Extract all features from a URL"""
        features = {}
        
        # Parse URL
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path
        except:
            return None
        
        # Basic length features
        features['length_url'] = len(url)
        features['length_hostname'] = len(domain)
        
        # Character count features
        features['nb_dots'] = url.count('.')
        features['nb_hyphens'] = url.count('-')
        features['nb_at'] = url.count('@')
        features['nb_and'] = url.count('&')
        features['nb_or'] = url.count('|')
        features['nb_underscore'] = url.count('_')
        features['nb_slash'] = url.count('/')
        features['nb_colon'] = url.count(':')
        features['nb_comma'] = url.count(',')
        features['nb_semicolumn'] = url.count(';')
        features['nb_dollar'] = url.count('$')
        features['nb_space'] = url.count(' ')
        features['nb_www'] = url.count('www')
        features['nb_com'] = url.count('.com')
        
        # Check if IP address is used
        features['ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0
        
        # Additional sophisticated features
        features['nb_percent'] = url.count('%')
        features['nb_question'] = url.count('?')
        features['nb_equal'] = url.count('=')
        features['nb_tilde'] = url.count('~')
        features['nb_digits'] = sum(c.isdigit() for c in url)
        features['nb_letters'] = sum(c.isalpha() for c in url)
        
        # Suspicious keyword count
        features['nb_suspicious_words'] = sum(
            word in url.lower() for word in self.suspicious_words
        )
        
        # URL complexity
        features['ratio_digits'] = features['nb_digits'] / (len(url) + 1)
        features['ratio_letters'] = features['nb_letters'] / (len(url) + 1)
        
        # Domain features
        features['domain_length'] = len(domain)
        features['nb_subdomains'] = domain.count('.') - 1 if domain.count('.') > 0 else 0
        
        # Path features
        features['path_length'] = len(path)
        features['nb_parameters'] = url.count('=')
        
        # HTTPS check
        features['is_https'] = 1 if url.startswith('https://') else 0
        
        # Shortened URL indicators
        features['is_shortened'] = 1 if any(x in domain.lower() for x in ['bit.ly', 'goo.gl', 'tinyurl', 't.co', 'ow.ly']) else 0
        
        # Placeholder for web traffic (will be populated from dataset)
        features['web_traffic'] = 0
        
        return features

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset"""
    print("=" * 60)
    print("STEP 1: LOADING DATASET")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_excel(file_path)
    print(f"✓ Dataset loaded successfully")
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {df.shape[1]}")
    print(f"\n  Class distribution:")
    print(f"  {df['status'].value_counts().to_dict()}")
    
    # Check for missing values
    print(f"\n✓ Missing values: {df.isnull().sum().sum()}")
    
    return df

def create_train_test_split(df):
    """Create train, validation, and test splits"""
    print("\n" + "=" * 60)
    print("STEP 2: CREATING TRAIN/VALIDATION/TEST SPLITS")
    print("=" * 60)
    
    # Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['status'])
    
    # Features and labels
    feature_columns = [col for col in df.columns if col not in ['url', 'status', 'label_encoded']]
    X = df[feature_columns].values
    y = df['label_encoded'].values
    urls = df['url'].values
    
    # Split: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test, urls_temp, urls_test = train_test_split(
        X, y, urls, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val, urls_train, urls_val = train_test_split(
        X_temp, y_temp, urls_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 of 0.85 ≈ 0.15 of total
    )
    
    print(f"✓ Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"✓ Validation set: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
    print(f"✓ Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
    
    # Class distribution
    print(f"\n  Training - Legitimate: {sum(y_train==1)}, Phishing: {sum(y_train==0)}")
    print(f"  Validation - Legitimate: {sum(y_val==1)}, Phishing: {sum(y_val==0)}")
    print(f"  Test - Legitimate: {sum(y_test==1)}, Phishing: {sum(y_test==0)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_columns, le

def scale_features(X_train, X_val, X_test):
    """Standardize features using StandardScaler"""
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE SCALING")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Features scaled using StandardScaler")
    print(f"  Mean: {X_train_scaled.mean():.6f}")
    print(f"  Std: {X_train_scaled.std():.6f}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, 
                          scaler, le, feature_columns):
    """Save all preprocessed data and objects"""
    print("\n" + "=" * 60)
    print("STEP 4: SAVING PREPROCESSED DATA")
    print("=" * 60)
    
    # Save datasets
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)
    print(f"✓ Saved training, validation, and test sets")
    
    # Save scaler and label encoder
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved StandardScaler")
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print(f"✓ Saved LabelEncoder")
    
    # Save feature names
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    print(f"✓ Saved feature column names")
    
    # Save metadata
    metadata = {
        'n_features': len(feature_columns),
        'feature_names': feature_columns,
        'classes': le.classes_.tolist(),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'legitimate_label': int(le.transform(['legitimate'])[0]),
        'phishing_label': int(le.transform(['phishing'])[0])
    }
    
    with open('preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✓ Saved preprocessing metadata")
    
    print(f"\n✓ All preprocessing artifacts saved successfully!")

def main():
    """Main preprocessing pipeline"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "PHISHING DETECTION - DATA PREPROCESSING" + " " * 9 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Load data
    df = load_and_preprocess_data('/mnt/user-data/uploads/DATASET_33_56_.xlsx')
    
    # Create splits
    X_train, X_val, X_test, y_train, y_val, y_test, feature_columns, le = create_train_test_split(df)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Save everything
    save_preprocessed_data(X_train_scaled, X_val_scaled, X_test_scaled, 
                          y_train, y_val, y_test, scaler, le, feature_columns)
    
    # Final summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"✓ Ready for model training")
    print(f"✓ Feature dimension: {X_train_scaled.shape[1]}")
    print(f"✓ Total samples processed: {len(X_train) + len(X_val) + len(X_test)}")
    print("\nNext step: Run 2_train_hybrid_model.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
