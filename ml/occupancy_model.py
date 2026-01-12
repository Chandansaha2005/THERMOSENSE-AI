"""
Occupancy Prediction Model
Predicts occupancy 30 minutes ahead
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from ml.features import FeatureEngineer


class OccupancyPredictor:
    """Predicts occupancy using RandomForest"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.feature_engineer = FeatureEngineer()
        self.trained = False
        self.feature_columns = None
        
    def prepare_data(self, df):
        """Prepare data for training"""
        df_features = self.feature_engineer.build_occupancy_features(df)
        self.feature_columns = self.feature_engineer.get_occupancy_feature_columns()
        
        missing_cols = set(self.feature_columns) - set(df_features.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        prediction_steps = int(PREDICTION_HORIZON / TIME_STEP_MINUTES)
        df_features['occupancy_future'] = df_features['occupancy'].shift(-prediction_steps)
        df_features = df_features.dropna(subset=['occupancy_future'])
        
        X = df_features[self.feature_columns].values
        y = df_features['occupancy_future'].values
        
        return X, y
    
    def train(self, df):
        """Train model"""
        print("Training Occupancy Prediction Model...")
        
        X, y = self.prepare_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"Training: {len(X_train)}, Test: {len(X_test)}")
        
        self.feature_engineer.fit_scaler(X_train)
        X_train_scaled = self.feature_engineer.transform(X_train)
        X_test_scaled = self.feature_engineer.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        self.trained = True
        
        train_pred = np.maximum(self.model.predict(X_train_scaled), 0)
        test_pred = np.maximum(self.model.predict(X_test_scaled), 0)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        print(f"✓ Train MAE: {metrics['train_mae']:.2f} persons")
        print(f"✓ Test MAE: {metrics['test_mae']:.2f} persons")
        print(f"✓ Test R²: {metrics['test_r2']:.3f}")
        
        return metrics
    
    def predict(self, current_state):
        """Predict occupancy"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        df_single = pd.DataFrame([current_state])
        df_features = self.feature_engineer.build_occupancy_features(df_single)
        
        if len(df_features) == 0:
            return current_state.get('occupancy', 0)
        
        X = df_features[self.feature_columns].iloc[-1:].values
        X_scaled = self.feature_engineer.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        
        return max(0, round(prediction))
    
    def save(self, path=None):
        """Save model"""
        if path is None:
            os.makedirs(MODEL_DIR, exist_ok=True)
            path = OCCUPANCY_MODEL_PATH
        
        model_data = {
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'feature_columns': self.feature_columns,
            'trained': self.trained
        }
        
        joblib.dump(model_data, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path=None):
        """Load model"""
        if path is None:
            path = OCCUPANCY_MODEL_PATH
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_engineer = model_data['feature_engineer']
        self.feature_columns = model_data['feature_columns']
        self.trained = model_data['trained']
        
        print(f"✓ Model loaded from {path}")
    
    def get_feature_importance(self, top_n=10):
        """Get feature importance"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)


if __name__ == "__main__":
    print("=== Occupancy Model Training ===\n")
    
    if not os.path.exists(SIMULATED_DATA_PATH):
        print("Error: Data not found. Run: python simulation/sensors.py")
        exit(1)
    
    df = pd.read_csv(SIMULATED_DATA_PATH)
    print(f"Loaded {len(df)} points\n")
    
    predictor = OccupancyPredictor()
    metrics = predictor.train(df)
    
    print("\n--- Top 10 Features ---")
    importance = predictor.get_feature_importance(10)
    print(importance.to_string(index=False))
    
    predictor.save()
    
    print("\n✓ Occupancy model complete")