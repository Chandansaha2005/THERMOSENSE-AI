"""
Temperature Prediction Model (FIXED VERSION)
Predicts temperature 30 minutes ahead with NaN handling
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from ml.features import FeatureEngineer


def validate_data(df, name="data"):
    """
    Validate data integrity
    Checks for NaN, inf, and reasonable temperature ranges
    """
    errors = []
    
    # Check for NaN values
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        errors.append(f"  ❌ NaN found in columns: {nan_cols}")
    
    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    inf_cols = numeric_df.columns[np.isinf(numeric_df).any()].tolist()
    if inf_cols:
        errors.append(f"  ❌ Inf found in columns: {inf_cols}")
    
    # Check temperature ranges
    if 'indoor_temp' in df.columns:
        temp_min = df['indoor_temp'].min()
        temp_max = df['indoor_temp'].max()
        if temp_min < 10.0 or temp_max > 50.0:
            errors.append(f"  ⚠️  Indoor temperature out of bounds: {temp_min:.2f}°C to {temp_max:.2f}°C (expected 16-45°C)")
    
    if 'outdoor_temp' in df.columns:
        temp_min = df['outdoor_temp'].min()
        temp_max = df['outdoor_temp'].max()
        if temp_min < 10.0 or temp_max > 60.0:
            errors.append(f"  ⚠️  Outdoor temperature out of bounds: {temp_min:.2f}°C to {temp_max:.2f}°C (expected 20-50°C)")
    
    if errors:
        print(f"[Validation] {name}:")
        for error in errors:
            print(error)
        if any("❌" in e for e in errors):
            raise ValueError(f"Invalid {name}: {errors[0]}")
    
    return True


class TemperaturePredictor:
    """Predicts temperature using GradientBoosting"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        self.feature_engineer = FeatureEngineer()
        self.trained = False
        self.feature_columns = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, df):
        """Prepare data for training with validation"""
        # Validate input data
        validate_data(df, "input data")
        
        # Drop rows with invalid temperature values
        initial_len = len(df)
        df = df.dropna(subset=['indoor_temp', 'outdoor_temp'])
        df = df[(df['indoor_temp'] >= 16.0) & (df['indoor_temp'] <= 45.0)]
        df = df[(df['outdoor_temp'] >= 15.0) & (df['outdoor_temp'] <= 55.0)]
        
        if len(df) < initial_len:
            print(f"  [Data] Dropped {initial_len - len(df)} rows with invalid temps")
        
        # Convert numeric columns to float64
        numeric_cols = ['indoor_temp', 'outdoor_temp', 'occupancy', 'hvac_state', 'hour', 'day_of_week']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)
        
        # Build features
        df_features = self.feature_engineer.build_temperature_features(df)
        self.feature_columns = self.feature_engineer.get_temperature_feature_columns()
        
        missing_cols = set(self.feature_columns) - set(df_features.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        prediction_steps = int(PREDICTION_HORIZON / TIME_STEP_MINUTES)
        df_features['temp_future'] = df_features['indoor_temp'].shift(-prediction_steps)
        df_features = df_features.dropna(subset=['temp_future'])
        
        X = df_features[self.feature_columns].values.astype(np.float64)
        y = df_features['temp_future'].values.astype(np.float64)
        
        # Ensure finite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=np.nanmean(y), posinf=np.nanmax(y), neginf=np.nanmin(y))
        
        # Final validation
        if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
            raise ValueError("X or y contains non-finite values after cleaning")
        
        return X, y
    
    def train(self, df):
        """Train model with robust error handling"""
        print("Training Temperature Prediction Model...")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            X, y = self.prepare_data(df)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            print(f"Training: {len(X_train)}, Test: {len(X_test)}")
            
            # Use StandardScaler with float64
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train).astype(np.float64)
            X_test_scaled = self.scaler.transform(X_test).astype(np.float64)
            
            # Ensure finite after scaling
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Verify before training
            assert np.all(np.isfinite(X_train_scaled)), "X_train_scaled contains non-finite values"
            assert np.all(np.isfinite(X_test_scaled)), "X_test_scaled contains non-finite values"
            assert np.all(np.isfinite(y_train)), "y_train contains non-finite values"
            assert np.all(np.isfinite(y_test)), "y_test contains non-finite values"
            
            # Train with warnings captured
            self.model.fit(X_train_scaled, y_train)
            self.trained = True
            
            # Log any numerical warnings
            for warning in w:
                if 'overflow' in str(warning.message).lower() or 'underflow' in str(warning.message).lower():
                    print(f"  ⚠️  {warning.message}")
            
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            metrics = {
                'train_mae': mean_absolute_error(y_train, train_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred)
            }
            
            print(f"✓ Train MAE: {metrics['train_mae']:.2f}°C")
            print(f"✓ Test MAE: {metrics['test_mae']:.2f}°C")
            print(f"✓ Test R²: {metrics['test_r2']:.3f}")
            
            return metrics
    
    def predict(self, current_state):
        """Predict temperature"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        df_single = pd.DataFrame([current_state])
        df_features = self.feature_engineer.build_temperature_features(df_single)
        
        if len(df_features) == 0:
            return current_state.get('indoor_temp', INITIAL_ROOM_TEMP)
        
        X = df_features[self.feature_columns].iloc[-1:].values.astype(np.float64)
        
        # Handle NaN in prediction input
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_scaled = self.scaler.transform(X).astype(np.float64)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        prediction = self.model.predict(X_scaled)[0]
        
        return float(np.clip(prediction, 16.0, 45.0))
    
    def save(self, path=None):
        """Save model"""
        if path is None:
            os.makedirs(MODEL_DIR, exist_ok=True)
            path = TEMPERATURE_MODEL_PATH
        
        model_data = {
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'trained': self.trained
        }
        
        joblib.dump(model_data, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path=None):
        """Load model"""
        if path is None:
            path = TEMPERATURE_MODEL_PATH
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_engineer = model_data['feature_engineer']
        self.feature_columns = model_data['feature_columns']
        self.scaler = model_data.get('scaler', StandardScaler())
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
    print("=== Temperature Model Training ===\n")
    
    if not os.path.exists(SIMULATED_DATA_PATH):
        print("❌ Error: Data not found. Run: python simulation/sensors.py")
        sys.exit(1)
    
    try:
        df = pd.read_csv(SIMULATED_DATA_PATH)
        print(f"Loaded {len(df)} points\n")
        
        # Validate data before training
        validate_data(df, "loaded CSV")
        
        predictor = TemperaturePredictor()
        metrics = predictor.train(df)
        
        print("\n--- Top 10 Features ---")
        importance = predictor.get_feature_importance(10)
        print(importance.to_string(index=False))
        
        predictor.save()
        
        print("\n✓ Temperature model complete")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)