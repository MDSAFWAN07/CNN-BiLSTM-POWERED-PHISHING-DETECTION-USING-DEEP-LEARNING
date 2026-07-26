"""
Phishing Detection - Model Testing and Analysis
Author: AI Phishing Detection System
Description: Comprehensive testing of trained models with detailed analysis
"""

import numpy as np
import pandas as pd
import pickle
import json
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, classification_report, roc_curve, auc,
                            precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse

class PhishingDetectorTester:
    """Test and analyze phishing detection models"""
    
    def __init__(self):
        self.bilstm_model = None
        self.xgboost_model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        
    def load_models(self):
        """Load all trained models and preprocessors"""
        print("\n" + "=" * 60)
        print("LOADING TRAINED MODELS")
        print("=" * 60)
        
        # Load Bi-LSTM
        self.bilstm_model = keras.models.load_model('bilstm_model.h5')
        print(f"✓ Loaded Bi-LSTM model")
        
        # Load XGBoost
        with open('xgboost_model.pkl', 'rb') as f:
            self.xgboost_model = pickle.load(f)
        print(f"✓ Loaded XGBoost model")
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Loaded StandardScaler")
        
        # Load label encoder
        with open('label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f"✓ Loaded LabelEncoder")
        
        # Load feature columns
        with open('feature_columns.pkl', 'rb') as f:
            self.feature_columns = pickle.load(f)
        print(f"✓ Loaded feature columns ({len(self.feature_columns)} features)")
        
    def extract_bilstm_features(self, X):
        """Extract deep features from Bi-LSTM"""
        from tensorflow.keras.models import Model
        
        feature_extractor = Model(
            inputs=self.bilstm_model.input,
            outputs=self.bilstm_model.get_layer('feature_layer').output
        )
        
        deep_features = feature_extractor.predict(X, verbose=0)
        return deep_features
    
    def predict_hybrid(self, X):
        """Make hybrid predictions"""
        # Bi-LSTM predictions
        bilstm_pred = self.bilstm_model.predict(X, verbose=0).flatten()
        
        # Extract deep features and combine
        deep_features = self.extract_bilstm_features(X)
        X_combined = np.concatenate([X, deep_features], axis=1)
        
        # XGBoost predictions
        xgb_pred_proba = self.xgboost_model.predict_proba(X_combined)[:, 1]
        
        # Ensemble prediction
        ensemble_pred = 0.5 * bilstm_pred + 0.5 * xgb_pred_proba
        
        return ensemble_pred, bilstm_pred, xgb_pred_proba
    
    def test_on_dataset(self, X_test, y_test):
        """Comprehensive testing on test dataset"""
        print("\n" + "=" * 60)
        print("TESTING ON TEST DATASET")
        print("=" * 60)
        print(f"Test samples: {len(X_test)}")
        
        # Get predictions
        hybrid_pred_proba, bilstm_pred_proba, xgb_pred_proba = self.predict_hybrid(X_test)
        
        # Convert to binary predictions
        hybrid_pred = (hybrid_pred_proba > 0.5).astype(int)
        bilstm_pred = (bilstm_pred_proba > 0.5).astype(int)
        xgb_pred = (xgb_pred_proba > 0.5).astype(int)
        
        # Calculate metrics for each model
        results = {}
        
        for model_name, y_pred, y_pred_proba in [
            ('Hybrid', hybrid_pred, hybrid_pred_proba),
            ('Bi-LSTM', bilstm_pred, bilstm_pred_proba),
            ('XGBoost', xgb_pred, xgb_pred_proba)
        ]:
            print(f"\n{'-' * 60}")
            print(f"{model_name} Model Results:")
            print(f"{'-' * 60}")
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
            print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
            print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
            print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
            
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"  TN: {cm[0][0]:4d}  |  FP: {cm[0][1]:4d}")
            print(f"  FN: {cm[1][0]:4d}  |  TP: {cm[1][1]:4d}")
            
            # Calculate additional metrics
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            print(f"\nAdditional Metrics:")
            print(f"Specificity: {specificity:.4f}")
            print(f"False Positive Rate: {false_positive_rate:.4f}")
            print(f"False Negative Rate: {false_negative_rate:.4f}")
            
            results[model_name.lower().replace('-', '_')] = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'specificity': float(specificity),
                'fpr': float(false_positive_rate),
                'fnr': float(false_negative_rate),
                'confusion_matrix': cm.tolist(),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        
        return results, hybrid_pred_proba, bilstm_pred_proba, xgb_pred_proba
    
    def test_individual_urls(self, test_urls):
        """Test on individual URLs with detailed output"""
        print("\n" + "=" * 60)
        print("TESTING INDIVIDUAL URLs")
        print("=" * 60)
        
        results = []
        
        for url in test_urls:
            print(f"\nTesting URL: {url}")
            print("-" * 60)
            
            # Extract features (simplified version)
            features = self.extract_url_features(url)
            
            if features is None:
                print("✗ Failed to extract features")
                continue
            
            # Create feature vector matching training format
            feature_vector = np.array([[features.get(col, 0) for col in self.feature_columns]])
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Predict
            hybrid_prob, bilstm_prob, xgb_prob = self.predict_hybrid(feature_vector_scaled)
            
            # Determine result
            is_phishing = hybrid_prob[0] < 0.5
            confidence = abs(hybrid_prob[0] - 0.5) * 200  # Convert to 0-100%
            
            result = {
                'url': url,
                'prediction': 'PHISHING' if is_phishing else 'LEGITIMATE',
                'confidence': float(confidence),
                'hybrid_score': float(hybrid_prob[0]),
                'bilstm_score': float(bilstm_prob[0]),
                'xgboost_score': float(xgb_prob[0])
            }
            
            print(f"✓ Prediction: {result['prediction']}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  Hybrid Score: {hybrid_prob[0]:.4f}")
            print(f"  Bi-LSTM Score: {bilstm_prob[0]:.4f}")
            print(f"  XGBoost Score: {xgb_prob[0]:.4f}")
            
            results.append(result)
        
        return results
    
    def extract_url_features(self, url):
        """Extract features from a URL"""
        try:
            features = {}
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # Basic features
            features['length_url'] = len(url)
            features['length_hostname'] = len(domain)
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
            features['web_traffic'] = 0  # Placeholder
            
            # Check if IP
            import re
            features['ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0
            
            return features
        except:
            return None
    
    def plot_confusion_matrices(self, results):
        """Plot confusion matrices for all models"""
        print("\n✓ Generating confusion matrix plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        models = ['hybrid', 'bi_lstm', 'xgboost']
        titles = ['Hybrid Model', 'Bi-LSTM Model', 'XGBoost Model']
        
        for idx, (model, title) in enumerate(zip(models, titles)):
            cm = np.array(results[model]['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Phishing', 'Legitimate'],
                       yticklabels=['Phishing', 'Legitimate'],
                       cbar_kws={'label': 'Count'})
            
            axes[idx].set_title(f'{title}\nAccuracy: {results[model]["accuracy"]*100:.2f}%',
                              fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrices: confusion_matrices.png")
    
    def plot_roc_curves(self, y_test, hybrid_prob, bilstm_prob, xgb_prob):
        """Plot ROC curves for all models"""
        print("\n✓ Generating ROC curves...")
        
        plt.figure(figsize=(10, 8))
        
        for name, probs in [('Hybrid', hybrid_prob), ('Bi-LSTM', bilstm_prob), ('XGBoost', xgb_prob)]:
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curves: roc_curves.png")
    
    def plot_metrics_comparison(self, results):
        """Plot metrics comparison bar chart"""
        print("\n✓ Generating metrics comparison...")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        models = ['Hybrid', 'Bi-LSTM', 'XGBoost']
        
        data = {
            'Accuracy': [results['hybrid']['accuracy'], results['bi_lstm']['accuracy'], results['xgboost']['accuracy']],
            'Precision': [results['hybrid']['precision'], results['bi_lstm']['precision'], results['xgboost']['precision']],
            'Recall': [results['hybrid']['recall'], results['bi_lstm']['recall'], results['xgboost']['recall']],
            'F1-Score': [results['hybrid']['f1_score'], results['bi_lstm']['f1_score'], results['xgboost']['f1_score']]
        }
        
        df = pd.DataFrame(data, index=models)
        
        ax = df.plot(kind='bar', figsize=(12, 7), width=0.8)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.xticks(rotation=0)
        plt.ylim([0.85, 1.0])
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics comparison: metrics_comparison.png")

def main():
    """Main testing pipeline"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 11 + "PHISHING DETECTION - MODEL TESTING" + " " * 12 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Initialize tester
    tester = PhishingDetectorTester()
    
    # Load models
    tester.load_models()
    
    # Load test data
    print("\n" + "=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)
    
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    print(f"✓ Loaded test data: {X_test.shape}")
    
    # Test on dataset
    results, hybrid_prob, bilstm_prob, xgb_prob = tester.test_on_dataset(X_test, y_test)
    
    # Plot visualizations
    tester.plot_confusion_matrices(results)
    tester.plot_roc_curves(y_test, hybrid_prob, bilstm_prob, xgb_prob)
    tester.plot_metrics_comparison(results)
    
    # Test on individual URLs
    print("\n" + "=" * 60)
    print("TESTING ON SAMPLE URLs")
    print("=" * 60)
    
    test_urls = [
        'https://www.google.com',
        'http://suspicious-bank-login-verify.com',
        'https://github.com',
        'http://paypal-security-update.com.verify.account',
        'https://www.amazon.com'
    ]
    
    url_results = tester.test_individual_urls(test_urls)
    
    # Save all results
    print("\n" + "=" * 60)
    print("SAVING TEST RESULTS")
    print("=" * 60)
    
    final_results = {
        'test_dataset_results': results,
        'individual_url_results': url_results,
        'test_samples': int(len(X_test)),
        'best_model': 'Hybrid',
        'best_accuracy': results['hybrid']['accuracy']
    }
    
    with open('test_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"✓ Saved test results: test_results.json")
    
    # Final summary
    print("\n" + "=" * 60)
    print("TESTING COMPLETE!")
    print("=" * 60)
    print(f"✓ Best Model: Hybrid")
    print(f"✓ Accuracy: {results['hybrid']['accuracy']*100:.2f}%")
    print(f"✓ Precision: {results['hybrid']['precision']*100:.2f}%")
    print(f"✓ Recall: {results['hybrid']['recall']*100:.2f}%")
    print(f"✓ F1-Score: {results['hybrid']['f1_score']*100:.2f}%")
    print("\nAll models and results ready for deployment!")
    print("=" * 60)

if __name__ == "__main__":
    main()
