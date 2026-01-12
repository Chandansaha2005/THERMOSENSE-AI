"""
Feature Engineering Pipeline
Creates features for ML models
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *


class FeatureEngineer:
    """Feature engineering for HVAC prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        
    def create_lag_features(self, df, column, lags=LAG_FEATURES):
        """Create lag features"""
        df_copy = df.copy()
        
        for lag in lags:
            periods = int(lag / TIME_STEP_MINUTES)
            lag_col_name = f'{column}_lag_{lag}min'
            df_copy[lag_col_name] = df_copy[column].shift(periods).astype(np.float64)
        
        return df_copy
    
    def create_time_features(self, df):
        """Create time-based features"""
        df_copy = df.copy()
        
        if 'timestamp' in df_copy.columns:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            df_copy['hour'] = df_copy['timestamp'].dt.hour.astype(np.float64)
            df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek.astype(np.float64)
        
        df_copy['hour_sin'] = np.sin(2.0 * np.pi * df_copy['hour'] / 24.0).astype(np.float64)
        df_copy['hour_cos'] = np.cos(2.0 * np.pi * df_copy['hour'] / 24.0).astype(np.float64)
        df_copy['day_sin'] = np.sin(2.0 * np.pi * df_copy['day_of_week'] / 7.0).astype(np.float64)
        df_copy['day_cos'] = np.cos(2.0 * np.pi * df_copy['day_of_week'] / 7.0).astype(np.float64)
        
        df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(np.float64)
        df_copy['is_office_hours'] = (
            (df_copy['hour'] >= OFFICE_START_HOUR) & 
            (df_copy['hour'] < OFFICE_END_HOUR)
        ).astype(np.float64)
        
        return df_copy
    
    def create_rolling_features(self, df, column, windows=[2, 4, 8]):
        """Create rolling statistics with NaN handling"""
        df_copy = df.copy()
        
        for window in windows:
            roll_col_name = f'{column}_rolling_mean_{window}'
            df_copy[roll_col_name] = df_copy[column].rolling(
                window=window, min_periods=1
            ).mean().astype(np.float64)
            
            # Fill any remaining NaN with column mean
            if df_copy[roll_col_name].isna().any():
                col_mean = df_copy[roll_col_name].mean()
                df_copy[roll_col_name] = df_copy[roll_col_name].fillna(col_mean)
        
        return df_copy
    
    def build_occupancy_features(self, df):
        """Build features for occupancy prediction"""
        df_features = self.create_time_features(df)
        df_features = self.create_lag_features(df_features, 'occupancy', [15, 30, 60])
        df_features = self.create_lag_features(df_features, 'indoor_temp', [15, 30])
        df_features = self.create_rolling_features(df_features, 'occupancy', [4, 8])
        df_features = df_features.dropna()
        
        # Ensure all numeric columns are float64
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_features[col] = df_features[col].astype(np.float64)
        
        return df_features
    
    def build_temperature_features(self, df):
        """Build features for temperature prediction"""
        df_features = self.create_time_features(df)
        df_features = self.create_lag_features(df_features, 'indoor_temp', [15, 30, 60])
        df_features = self.create_lag_features(df_features, 'outdoor_temp', [15, 30])
        df_features = self.create_lag_features(df_features, 'occupancy', [15, 30])
        df_features = self.create_lag_features(df_features, 'hvac_state', [15, 30])
        
        df_features['temp_trend'] = (df_features['indoor_temp'] - df_features['indoor_temp_lag_15min']).astype(np.float64)
        df_features['temp_diff'] = (df_features['outdoor_temp'] - df_features['indoor_temp']).astype(np.float64)
        
        df_features = df_features.dropna()
        
        # Ensure all numeric columns are float64
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_features[col] = df_features[col].astype(np.float64)
        
        # Final NaN check
        if df_features.isna().any().any():
            df_features = df_features.fillna(df_features.mean(numeric_only=True))
        
        return df_features
    
    def get_occupancy_feature_columns(self):
        """Return occupancy feature columns"""
        return [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_weekend', 'is_office_hours',
            'occupancy_lag_15min', 'occupancy_lag_30min', 'occupancy_lag_60min',
            'indoor_temp_lag_15min', 'indoor_temp_lag_30min',
            'occupancy_rolling_mean_4', 'occupancy_rolling_mean_8'
        ]
    
    def get_temperature_feature_columns(self):
        """Return temperature feature columns"""
        return [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_weekend', 'is_office_hours',
            'indoor_temp_lag_15min', 'indoor_temp_lag_30min', 'indoor_temp_lag_60min',
            'outdoor_temp_lag_15min', 'outdoor_temp_lag_30min',
            'occupancy_lag_15min', 'occupancy_lag_30min',
            'hvac_state_lag_15min', 'hvac_state_lag_30min',
            'temp_trend', 'temp_diff',
            'outdoor_temp', 'occupancy', 'hvac_state'
        ]
    
    def fit_scaler(self, X):
        """Fit scaler on training data"""
        # Ensure X is float64
        X = np.asarray(X, dtype=np.float64)
        self.scaler.fit(X)
        self.fitted = True
    
    def transform(self, X):
        """Transform features"""
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        # Ensure X is float64
        X = np.asarray(X, dtype=np.float64)
        X_transformed = self.scaler.transform(X).astype(np.float64)
        # Replace any NaN/inf with 0
        X_transformed = np.nan_to_num(X_transformed, nan=0.0, posinf=0.0, neginf=0.0)
        return X_transformed
    
    def fit_transform(self, X):
        """Fit and transform"""
        self.fit_scaler(X)
        return self.transform(X)


if __name__ == "__main__":
    print("Testing Feature Engineering\n")
    
    if os.path.exists(SIMULATED_DATA_PATH):
        df = pd.read_csv(SIMULATED_DATA_PATH)
        print(f"Loaded {len(df)} rows")
    else:
        print("Error: Run sensors.py first")
        exit(1)
    
    fe = FeatureEngineer()
    
    print("\n--- Occupancy Features ---")
    df_occ = fe.build_occupancy_features(df)
    occ_features = fe.get_occupancy_feature_columns()
    print(f"Created {len(occ_features)} features")
    print(f"Data shape: {df_occ.shape}")
    
    print("\n--- Temperature Features ---")
    df_temp = fe.build_temperature_features(df)
    temp_features = fe.get_temperature_feature_columns()
    print(f"Created {len(temp_features)} features")
    print(f"Data shape: {df_temp.shape}")
    
    print("\n✓ Feature engineering working")