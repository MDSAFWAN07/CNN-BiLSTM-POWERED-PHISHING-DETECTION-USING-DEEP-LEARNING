"""
Phishing Detection - Hybrid Bi-LSTM + XGBoost Model Training
Author: AI Phishing Detection System
Description: Train hybrid deep learning model for phishing detection
"""

import numpy as np
import pandas as pd
import pickle
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class HybridPhishingDetector:
    """Hybrid Bi-LSTM + XGBoost model for phishing detection"""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.bilstm_model = None
        self.xgboost_model = None
        self.history = None
        
    def build_bilstm_model(self):
        """Build Bi-LSTM model architecture"""
        print("\n" + "=" * 60)
        print("BUILDING BI-LSTM MODEL ARCHITECTURE")
        print("=" * 60)
        
        model = Sequential([
            # Reshape for LSTM (batch_size, timesteps, features)
            keras.layers.Reshape((self.input_dim, 1), input_shape=(self.input_dim,)),
            
            # First Bi-LSTM layer
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
            BatchNormalization(),
            
            # Second Bi-LSTM layer
            Bidirectional(LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2)),
            BatchNormalization(),
            
            # Dense layers
            Dense(128, activation='relu'),
            Dropout(0.4),
            BatchNormalization(),
            
            Dense(64, activation='relu'),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            # Output layer for feature extraction (not final classification)
            Dense(16, activation='relu', name='feature_layer'),
            
            # Final classification layer
            Dense(1, activation='sigmoid', name='output')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        print(f"✓ Bi-LSTM model built successfully")
        print(f"\nModel Summary:")
        model.summary()
        
        self.bilstm_model = model
        return model
    
    def train_bilstm(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """Train Bi-LSTM model"""
        print("\n" + "=" * 60)
        print("TRAINING BI-LSTM MODEL")
        print("=" * 60)
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'best_bilstm_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        print(f"✓ Training configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
        # Train model
        history = self.bilstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )
        
        self.history = history
        print(f"\n✓ Bi-LSTM training completed!")
        
        return history
    
    def extract_bilstm_features(self, X):
        """Extract deep features from Bi-LSTM for XGBoost"""
        # Create feature extraction model (up to feature_layer)
        feature_extractor = Model(
            inputs=self.bilstm_model.input,
            outputs=self.bilstm_model.get_layer('feature_layer').output
        )
        
        # Extract features
        deep_features = feature_extractor.predict(X, verbose=0)
        return deep_features
    
    def train_xgboost(self, X_train_orig, y_train, X_val_orig, y_val):
        """Train XGBoost on original features + Bi-LSTM features"""
        print("\n" + "=" * 60)
        print("TRAINING XGBOOST MODEL")
        print("=" * 60)
        
        # Extract deep features from Bi-LSTM
        print("✓ Extracting deep features from Bi-LSTM...")
        train_deep_features = self.extract_bilstm_features(X_train_orig)
        val_deep_features = self.extract_bilstm_features(X_val_orig)
        
        # Combine original features with deep features
        X_train_combined = np.concatenate([X_train_orig, train_deep_features], axis=1)
        X_val_combined = np.concatenate([X_val_orig, val_deep_features], axis=1)
        
        print(f"✓ Combined feature dimension: {X_train_combined.shape[1]}")
        print(f"  Original features: {X_train_orig.shape[1]}")
        print(f"  Deep features: {train_deep_features.shape[1]}")
        
        # Train XGBoost
        print("\n✓ Training XGBoost classifier...")
        self.xgboost_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=15
        )
        
        self.xgboost_model.fit(
            X_train_combined, y_train,
            eval_set=[(X_val_combined, y_val)],
            verbose=True
        )
        
        print(f"\n✓ XGBoost training completed!")
        
        return X_train_combined, X_val_combined
    
    def predict_hybrid(self, X):
        """Hybrid prediction using both models"""
        # Get Bi-LSTM predictions
        bilstm_pred = self.bilstm_model.predict(X, verbose=0)
        
        # Extract deep features
        deep_features = self.extract_bilstm_features(X)
        
        # Combine original + deep features
        X_combined = np.concatenate([X, deep_features], axis=1)
        
        # Get XGBoost predictions
        xgb_pred_proba = self.xgboost_model.predict_proba(X_combined)[:, 1]
        
        # Ensemble: weighted average
        ensemble_pred = 0.5 * bilstm_pred.flatten() + 0.5 * xgb_pred_proba
        
        return ensemble_pred
    
    def save_models(self):
        """Save both models"""
        print("\n" + "=" * 60)
        print("SAVING MODELS")
        print("=" * 60)
        
        # Save Bi-LSTM model
        self.bilstm_model.save('bilstm_model.h5')
        print(f"✓ Saved Bi-LSTM model: bilstm_model.h5")
        
        # Save XGBoost model
        with open('xgboost_model.pkl', 'wb') as f:
            pickle.dump(self.xgboost_model, f)
        print(f"✓ Saved XGBoost model: xgboost_model.pkl")
        
        # Save hybrid model wrapper
        hybrid_info = {
            'input_dim': self.input_dim,
            'bilstm_model': 'bilstm_model.h5',
            'xgboost_model': 'xgboost_model.pkl',
            'ensemble_weights': {'bilstm': 0.5, 'xgboost': 0.5}
        }
        
        with open('hybrid_model_info.json', 'w') as f:
            json.dump(hybrid_info, f, indent=4)
        print(f"✓ Saved hybrid model configuration")

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance"""
    print("\n" + "=" * 60)
    print(f"EVALUATING {model_name}")
    print("=" * 60)
    
    # Predictions
    if model_name == "Hybrid":
        y_pred_proba = model.predict_hybrid(X_test)
    elif model_name == "Bi-LSTM":
        y_pred_proba = model.bilstm_model.predict(X_test, verbose=0).flatten()
    else:  # XGBoost
        deep_features = model.extract_bilstm_features(X_test)
        X_test_combined = np.concatenate([X_test, deep_features], axis=1)
        y_pred_proba = model.xgboost_model.predict_proba(X_test_combined)[:, 1]
    
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n✓ Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n✓ Confusion Matrix:")
    print(f"  TN: {cm[0][0]:4d}  |  FP: {cm[0][1]:4d}")
    print(f"  FN: {cm[1][0]:4d}  |  TP: {cm[1][1]:4d}")
    
    # Classification Report
    print(f"\n✓ Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Phishing', 'Legitimate']))
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }

def plot_training_history(history):
    """Plot training history"""
    print("\n✓ Generating training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
    axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
    axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved training plots: training_history.png")

def main():
    """Main training pipeline"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 8 + "PHISHING DETECTION - HYBRID MODEL TRAINING" + " " * 8 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Load preprocessed data
    print("=" * 60)
    print("LOADING PREPROCESSED DATA")
    print("=" * 60)
    
    X_train = np.load('X_train.npy')
    X_val = np.load('X_val.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')
    
    print(f"✓ Loaded all datasets")
    print(f"  Train: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Initialize hybrid model
    input_dim = X_train.shape[1]
    hybrid_model = HybridPhishingDetector(input_dim)
    
    # Build and train Bi-LSTM
    hybrid_model.build_bilstm_model()
    history = hybrid_model.train_bilstm(X_train, y_train, X_val, y_val, epochs=50, batch_size=64)
    
    # Plot training history
    plot_training_history(history)
    
    # Train XGBoost on combined features
    X_train_combined, X_val_combined = hybrid_model.train_xgboost(X_train, y_train, X_val, y_val)
    
    # Evaluate all models
    bilstm_metrics = evaluate_model(hybrid_model, X_test, y_test, "Bi-LSTM")
    xgboost_metrics = evaluate_model(hybrid_model, X_test, y_test, "XGBoost")
    hybrid_metrics = evaluate_model(hybrid_model, X_test, y_test, "Hybrid")
    
    # Save models
    hybrid_model.save_models()
    
    # Save evaluation results
    results = {
        'bilstm_metrics': bilstm_metrics,
        'xgboost_metrics': xgboost_metrics,
        'hybrid_metrics': hybrid_metrics,
        'best_model': 'Hybrid',
        'test_samples': int(len(X_test))
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"✓ Models saved successfully")
    print(f"✓ Best Hybrid Model Accuracy: {hybrid_metrics['accuracy']*100:.2f}%")
    print("\nNext step: Run 3_test_model.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
