"""
Phishing URL Detection - Hybrid Model Training
Bi-LSTM + XGBoost Ensemble Approach
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

class HybridPhishingDetector:
    """Hybrid model combining Bi-LSTM and XGBoost"""
    
    def __init__(self, input_dim, lstm_units=128, xgb_params=None):
        self.input_dim = input_dim
        self.lstm_units = lstm_units
        self.bilstm_model = None
        self.xgboost_model = None
        self.ensemble_weights = {'bilstm': 0.6, 'xgboost': 0.4}  # Tunable
        
        # Default XGBoost parameters
        self.xgb_params = xgb_params or {
            'max_depth': 7,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    def build_bilstm_model(self):
        """Build Bi-LSTM model"""
        print("\nBuilding Bi-LSTM model...")
        
        model = Sequential([
            Input(shape=(1, self.input_dim)),
            Bidirectional(LSTM(self.lstm_units, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
        )
        
        print(model.summary())
        self.bilstm_model = model
        return model
    
    def train_bilstm(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """Train Bi-LSTM model"""
        print("\n" + "="*60)
        print("TRAINING BI-LSTM MODEL")
        print("="*60)
        
        # Reshape for LSTM (samples, timesteps, features)
        X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val_lstm = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-7),
            ModelCheckpoint('/home/claude/best_bilstm_model.h5', monitor='val_accuracy', 
                          save_best_only=True, verbose=1)
        ]
        
        # Train
        history = self.bilstm_model.fit(
            X_train_lstm, y_train,
            validation_data=(X_val_lstm, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Bi-LSTM training completed!")
        return history
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        self.xgboost_model = xgb.XGBClassifier(**self.xgb_params)
        
        # Train with validation set for early stopping
        self.xgboost_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        
        print("\n✓ XGBoost training completed!")
        return self.xgboost_model
    
    def predict_ensemble(self, X):
        """Make ensemble predictions"""
        # Reshape for LSTM
        X_lstm = X.reshape(X.shape[0], 1, X.shape[1])
        
        # Get predictions from both models
        bilstm_probs = self.bilstm_model.predict(X_lstm, verbose=0).flatten()
        xgb_probs = self.xgboost_model.predict_proba(X)[:, 1]
        
        # Weighted ensemble
        ensemble_probs = (self.ensemble_weights['bilstm'] * bilstm_probs + 
                         self.ensemble_weights['xgboost'] * xgb_probs)
        
        return ensemble_probs
    
    def evaluate(self, X_test, y_test, model_type='ensemble'):
        """Evaluate model performance"""
        if model_type == 'bilstm':
            X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            y_probs = self.bilstm_model.predict(X_test_lstm, verbose=0).flatten()
        elif model_type == 'xgboost':
            y_probs = self.xgboost_model.predict_proba(X_test)[:, 1]
        else:  # ensemble
            y_probs = self.predict_ensemble(X_test)
        
        y_pred = (y_probs >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_probs)
        }
        
        return metrics, y_pred, y_probs

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/claude/training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history plot saved!")

def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    filename = title.lower().replace(' ', '_') + '.png'
    plt.savefig(f'/home/claude/{filename}', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {filename}")

def main():
    """Main training pipeline"""
    print("="*60)
    print("PHISHING DETECTION - HYBRID MODEL TRAINING")
    print("Bi-LSTM + XGBoost Ensemble")
    print("="*60)
    
    # Load preprocessed data
    print("\n1. Loading preprocessed data...")
    X_train = np.load('/home/claude/X_train.npy')
    X_test = np.load('/home/claude/X_test.npy')
    y_train = np.load('/home/claude/y_train.npy')
    y_test = np.load('/home/claude/y_test.npy')
    print(f"   ✓ Training data: {X_train.shape}")
    print(f"   ✓ Test data: {X_test.shape}")
    
    # Split training data for validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    print(f"   ✓ Validation data: {X_val.shape}")
    
    # Initialize hybrid model
    print("\n2. Initializing hybrid model...")
    input_dim = X_train.shape[1]
    hybrid_model = HybridPhishingDetector(input_dim=input_dim, lstm_units=128)
    print(f"   ✓ Input dimension: {input_dim} features")
    
    # Build and train Bi-LSTM
    print("\n3. Training Bi-LSTM component...")
    hybrid_model.build_bilstm_model()
    history = hybrid_model.train_bilstm(X_train_split, y_train_split, X_val, y_val, 
                                       epochs=50, batch_size=64)
    
    # Plot training history
    plot_training_history(history)
    
    # Train XGBoost
    print("\n4. Training XGBoost component...")
    hybrid_model.train_xgboost(X_train_split, y_train_split, X_val, y_val)
    
    # Evaluate all models
    print("\n5. Evaluating models...")
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    # Bi-LSTM evaluation
    print("\n[Bi-LSTM Model]")
    bilstm_metrics, bilstm_pred, _ = hybrid_model.evaluate(X_test, y_test, 'bilstm')
    for metric, value in bilstm_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    plot_confusion_matrix(y_test, bilstm_pred, "Bi-LSTM Confusion Matrix")
    
    # XGBoost evaluation
    print("\n[XGBoost Model]")
    xgb_metrics, xgb_pred, _ = hybrid_model.evaluate(X_test, y_test, 'xgboost')
    for metric, value in xgb_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    plot_confusion_matrix(y_test, xgb_pred, "XGBoost Confusion Matrix")
    
    # Ensemble evaluation
    print("\n[Ensemble Model (Bi-LSTM + XGBoost)]")
    ensemble_metrics, ensemble_pred, ensemble_probs = hybrid_model.evaluate(X_test, y_test, 'ensemble')
    for metric, value in ensemble_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    plot_confusion_matrix(y_test, ensemble_pred, "Ensemble Confusion Matrix")
    
    # Save models
    print("\n6. Saving models...")
    hybrid_model.bilstm_model.save('/home/claude/bilstm_model.h5')
    hybrid_model.xgboost_model.save_model('/home/claude/xgboost_model.json')
    print("   ✓ Bi-LSTM model saved: bilstm_model.h5")
    print("   ✓ XGBoost model saved: xgboost_model.json")
    
    # Save model metadata
    model_metadata = {
        'model_type': 'Hybrid (Bi-LSTM + XGBoost)',
        'input_features': int(input_dim),
        'bilstm_architecture': {
            'lstm_units': [128, 64],
            'dense_units': [64, 32],
            'dropout': [0.3, 0.3, 0.2, 0.2]
        },
        'xgboost_params': hybrid_model.xgb_params,
        'ensemble_weights': hybrid_model.ensemble_weights,
        'performance': {
            'bilstm': {k: float(v) for k, v in bilstm_metrics.items()},
            'xgboost': {k: float(v) for k, v in xgb_metrics.items()},
            'ensemble': {k: float(v) for k, v in ensemble_metrics.items()}
        },
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test))
    }
    
    with open('/home/claude/model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=4)
    print("   ✓ Model metadata saved: model_metadata.json")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n✓ Best Model: Ensemble")
    print(f"  • Accuracy: {ensemble_metrics['accuracy']:.4f}")
    print(f"  • Precision: {ensemble_metrics['precision']:.4f}")
    print(f"  • Recall: {ensemble_metrics['recall']:.4f}")
    print(f"  • F1-Score: {ensemble_metrics['f1_score']:.4f}")
    print(f"  • ROC-AUC: {ensemble_metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
