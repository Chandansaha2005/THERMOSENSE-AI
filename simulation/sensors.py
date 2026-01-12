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
    """Simulates indoor temperature with thermal dynamics"""
    
    def __init__(self, initial_temp=INITIAL_ROOM_TEMP):
        self.temperature = initial_temp
        
    def update(self, outdoor_temp, occupancy, hvac_state, pcm_cooling_kw, time_step_hours):
        """Update temperature based on heat balance"""
        occupancy_heat_kw = (occupancy * HEAT_PER_PERSON) / 1000.0
        temp_diff = outdoor_temp - self.temperature
        outdoor_heat_kw = HEAT_TRANSFER_COEFF * ROOM_SURFACE_AREA * temp_diff / 1000.0
        hvac_cooling_kw = -HVAC_COOLING_CAPACITY if hvac_state == 1 else 0.0
        pcm_cooling_contribution = -pcm_cooling_kw
        
        total_heat_kw = (occupancy_heat_kw + outdoor_heat_kw + 
                        hvac_cooling_kw + pcm_cooling_contribution)
        
        room_mass = ROOM_VOLUME * AIR_DENSITY
        heat_capacity = room_mass * AIR_SPECIFIC_HEAT
        energy_j = total_heat_kw * time_step_hours * 3600 * 1000
        temp_change = energy_j / heat_capacity
        
        self.temperature += temp_change
        self.temperature += np.random.normal(0, 0.1)
        
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
    
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(SIMULATED_DATA_PATH, index=False)
    
    print(f"✓ Generated {len(df)} data points")
    print(f"✓ Saved to {SIMULATED_DATA_PATH}")
    print(f"\nData Summary:")
    print(df.describe())