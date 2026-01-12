"""
Simulated IoT Sensors for THERMOSENSE-AI
Generates realistic sensor data with patterns and noise
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *


class TemperatureSensor:
    """Simulates outdoor temperature with daily patterns"""
    
    def __init__(self, base_temp=33.0):
        self.base_temp = base_temp
        
    def read(self, hour, day_of_week):
        """Generate outdoor temperature based on time"""
        daily_variation = 5 * np.sin((hour - 6) * np.pi / 12)
        noise = np.random.normal(0, 1.5)
        weekend_factor = 1.0 if day_of_week < 5 else 2.0
        temp = self.base_temp + daily_variation + noise + weekend_factor
        return np.clip(temp, OUTDOOR_TEMP_MIN, OUTDOOR_TEMP_MAX)


class OccupancySensor:
    """Simulates occupancy with office patterns"""
    
    def __init__(self):
        self.max_occupancy = MAX_OCCUPANCY
        
    def read(self, hour, day_of_week):
        """Generate occupancy count"""
        if day_of_week >= 5:  # Weekend
            if 10 <= hour <= 14:
                return np.random.randint(0, 3)
            return 0
        
        # Weekday
        if OFFICE_START_HOUR <= hour < OFFICE_END_HOUR:
            if 10 <= hour <= 16:  # Peak hours
                return np.random.randint(6, self.max_occupancy + 1)
            return np.random.randint(2, 7)
        elif 12 <= hour < 13:  # Lunch
            return np.random.randint(0, 4)
        return 0


class HVACStateSensor:
    """Tracks HVAC state"""
    
    def __init__(self):
        self.current_state = 0
        
    def set_state(self, state):
        self.current_state = int(state)
        
    def read(self):
        return self.current_state


class IndoorTemperatureSensor:
    """Simulates indoor temperature with bounded first-order thermal model"""
    
    def __init__(self, initial_temp=INITIAL_ROOM_TEMP):
        self.temperature = float(np.clip(initial_temp, 16.0, 45.0))
        # First-order thermal time constant (hours)
        self.tau = 0.5  # 30-minute time constant for realistic responsiveness
        
    def update(self, outdoor_temp, occupancy, hvac_state, pcm_cooling_kw, time_step_hours):
        """Update temperature using bounded first-order thermal model"""
        # Explicit float casting for all inputs
        outdoor_temp = float(outdoor_temp)
        occupancy = float(occupancy)
        hvac_state = float(hvac_state)
        pcm_cooling_kw = float(pcm_cooling_kw)
        time_step_hours = float(time_step_hours)
        current_temp = float(self.temperature)
        
        # Calculate heat gains in °C/hour equivalent
        occupancy_heat_effect = occupancy * HEAT_PER_PERSON / 100.0  # ~0.1-1°C per person
        
        # Cooling effects
        hvac_cooling_effect = float(HVAC_COOLING_CAPACITY) if hvac_state == 1.0 else 0.0
        hvac_cooling_effect *= -0.8  # Reduces indoor temp by up to 0.8°C per hour
        
        pcm_cooling_effect = float(pcm_cooling_kw) * -0.5  # PCM contribution
        
        # Outdoor influence (first-order approach to equilibrium)
        temp_diff = outdoor_temp - current_temp
        # Bounded heat transfer (prevent extreme changes)
        outdoor_influence = float(np.clip(temp_diff * 0.15, -2.0, 2.0))  # Max ±2°C/hour
        
        # First-order thermal dynamics: dT/dt = (T_equilibrium - T) / tau + noise
        equilibrium_temp = current_temp + outdoor_influence + occupancy_heat_effect + hvac_cooling_effect + pcm_cooling_effect
        
        # Apply first-order filter
        alpha = float(np.clip(time_step_hours / self.tau, 0.0, 1.0))
        new_temp = current_temp + alpha * (equilibrium_temp - current_temp)
        
        # Add bounded measurement noise
        noise = float(np.random.normal(0, 0.2))
        noise = float(np.clip(noise, -0.5, 0.5))
        new_temp += noise
        
        # Clamp to physical bounds
        self.temperature = float(np.clip(new_temp, 16.0, 45.0))
        
        return self.temperature
    
    def read(self):
        return self.temperature


def generate_simulation_data(hours=SIMULATION_HOURS):
    """Generate complete simulation dataset"""
    outdoor_temp_sensor = TemperatureSensor()
    occupancy_sensor = OccupancySensor()
    indoor_temp_sensor = IndoorTemperatureSensor()
    hvac_sensor = HVACStateSensor()
    
    time_steps = int(hours * 60 / TIME_STEP_MINUTES)
    start_time = datetime(2024, 1, 1, 0, 0)
    data = []
    
    for step in range(time_steps):
        current_time = start_time + timedelta(minutes=step * TIME_STEP_MINUTES)
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        outdoor_temp = outdoor_temp_sensor.read(hour, day_of_week)
        occupancy = occupancy_sensor.read(hour, day_of_week)
        indoor_temp = indoor_temp_sensor.read()
        
        if indoor_temp > COMFORT_MAX:
            hvac_sensor.set_state(1)
        elif indoor_temp < COMFORT_MIN:
            hvac_sensor.set_state(0)
        hvac_state = hvac_sensor.read()
        
        time_step_hours = TIME_STEP_MINUTES / 60.0
        new_indoor_temp = indoor_temp_sensor.update(
            outdoor_temp, occupancy, hvac_state, 0.0, time_step_hours
        )
        
        data.append({
            'timestamp': current_time,
            'hour': hour,
            'day_of_week': day_of_week,
            'outdoor_temp': outdoor_temp,
            'indoor_temp': new_indoor_temp,
            'occupancy': occupancy,
            'hvac_state': hvac_state
        })
    
    df = pd.DataFrame(data)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_office_hours'] = ((df['hour'] >= OFFICE_START_HOUR) & 
                              (df['hour'] < OFFICE_END_HOUR)).astype(int)
    
    return df


if __name__ == "__main__":
    print("Generating simulated sensor data...")
    df = generate_simulation_data()
    
    # Data validation and cleaning before saving
    print("\n[Validation] Checking data integrity...")
    
    # Replace NaN and inf values
    numeric_cols = ['outdoor_temp', 'indoor_temp', 'occupancy', 'hour', 'day_of_week']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
            df[col] = np.nan_to_num(df[col], nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip temperature values to physical bounds
    df['indoor_temp'] = np.clip(df['indoor_temp'], 16.0, 45.0)
    df['outdoor_temp'] = np.clip(df['outdoor_temp'], 20.0, 50.0)
    
    # Validate no NaN or inf remain
    for col in numeric_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum()
            if nan_count > 0 or inf_count > 0:
                raise ValueError(f"Column '{col}' contains {nan_count} NaN and {inf_count} inf values")
    
    # Create data directory and save
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(SIMULATED_DATA_PATH, index=False)
    
    print(f"✓ Generated {len(df)} data points")
    print(f"✓ Saved to {SIMULATED_DATA_PATH}")
    print(f"\nData Summary:")
    print(df[['outdoor_temp', 'indoor_temp', 'occupancy']].describe())
    print(f"\nValidation:")
    print(f"  Indoor temp range: {df['indoor_temp'].min():.2f}°C to {df['indoor_temp'].max():.2f}°C")
    print(f"  Outdoor temp range: {df['outdoor_temp'].min():.2f}°C to {df['outdoor_temp'].max():.2f}°C")