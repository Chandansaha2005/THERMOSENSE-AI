"""
HVAC Environment Simulator
Integrates sensors, PCM, and thermal dynamics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from simulation.sensors import (
    TemperatureSensor, OccupancySensor, 
    IndoorTemperatureSensor, HVACStateSensor
)
from simulation.pcm import PCMStorage


class HVACEnvironment:
    """Complete HVAC environment simulation"""
    
    def __init__(self):
        self.outdoor_temp_sensor = TemperatureSensor()
        self.occupancy_sensor = OccupancySensor()
        self.indoor_temp_sensor = IndoorTemperatureSensor()
        self.hvac_sensor = HVACStateSensor()
        self.pcm = PCMStorage()
        
        self.total_hvac_energy = 0.0
        self.total_pcm_cooling_delivered = 0.0
        self.time_step_hours = TIME_STEP_MINUTES / 60.0
        self.history = []
        
    def step(self, current_time, hvac_action, pcm_action):
        """Advance environment by one time step"""
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        outdoor_temp = self.outdoor_temp_sensor.read(hour, day_of_week)
        occupancy = self.occupancy_sensor.read(hour, day_of_week)
        
        hvac_state = 1 if hvac_action == 'ON' else 0
        self.hvac_sensor.set_state(hvac_state)
        
        pcm_cooling_kw = 0.0
        
        if pcm_action == 'CHARGE' and self.pcm.can_charge():
            energy_charged = self.pcm.charge(self.time_step_hours)
            
        elif pcm_action == 'DISCHARGE' and self.pcm.can_discharge():
            pcm_cooling_kw = self.pcm.discharge(
                self.time_step_hours, 
                requested_cooling_kw=PCM_DISCHARGE_RATE
            )
            self.total_pcm_cooling_delivered += pcm_cooling_kw * self.time_step_hours
        
        indoor_temp = self.indoor_temp_sensor.update(
            outdoor_temp, 
            occupancy, 
            hvac_state, 
            pcm_cooling_kw,
            self.time_step_hours
        )
        
        if hvac_state == 1:
            hvac_energy = (HVAC_COOLING_CAPACITY / HVAC_COP) * self.time_step_hours
            self.total_hvac_energy += hvac_energy
        
        electricity_cost = self.get_electricity_cost(hour)
        
        state = {
            'timestamp': current_time,
            'hour': hour,
            'day_of_week': day_of_week,
            'outdoor_temp': outdoor_temp,
            'indoor_temp': indoor_temp,
            'occupancy': occupancy,
            'hvac_state': hvac_state,
            'hvac_action': hvac_action,
            'pcm_action': pcm_action,
            'pcm_charge_kwh': self.pcm.get_state_of_charge(),
            'pcm_charge_percent': self.pcm.get_state_of_charge_percent(),
            'pcm_cooling_kw': pcm_cooling_kw,
            'electricity_cost_per_kwh': electricity_cost
        }
        
        self.history.append(state)
        return state
    
    def get_electricity_cost(self, hour):
        """Get electricity cost by hour"""
        if 14 <= hour < 18:
            return PEAK_ELECTRICITY_COST
        elif 6 <= hour < 22:
            return STANDARD_ELECTRICITY_COST
        else:
            return OFF_PEAK_ELECTRICITY_COST
    
    def get_current_state(self):
        """Return current state"""
        return {
            'indoor_temp': self.indoor_temp_sensor.read(),
            'pcm_charge_kwh': self.pcm.get_state_of_charge(),
            'pcm_charge_percent': self.pcm.get_state_of_charge_percent(),
            'total_hvac_energy': self.total_hvac_energy
        }
    
    def reset(self):
        """Reset environment"""
        self.indoor_temp_sensor = IndoorTemperatureSensor()
        self.hvac_sensor = HVACStateSensor()
        self.pcm.reset()
        self.total_hvac_energy = 0.0
        self.total_pcm_cooling_delivered = 0.0
        self.history = []
    
    def get_history_df(self):
        """Return history as DataFrame"""
        return pd.DataFrame(self.history)
    
    def get_total_cost(self):
        """Calculate total cost"""
        df = self.get_history_df()
        if len(df) == 0:
            return 0.0
        
        df['hvac_energy_kwh'] = df['hvac_state'] * (HVAC_COOLING_CAPACITY / HVAC_COP) * self.time_step_hours
        df['cost'] = df['hvac_energy_kwh'] * df['electricity_cost_per_kwh']
        
        return df['cost'].sum()


class BaselineHVACController:
    """Traditional reactive controller"""
    
    def __init__(self, setpoint=25.0, deadband=0.5):
        self.setpoint = setpoint
        self.deadband = deadband
        self.hvac_state = 'OFF'
    
    def decide(self, indoor_temp):
        """Make decision based on temperature"""
        if indoor_temp > self.setpoint + self.deadband:
            self.hvac_state = 'ON'
        elif indoor_temp < self.setpoint - self.deadband:
            self.hvac_state = 'OFF'
        
        return self.hvac_state, 'IDLE'


if __name__ == "__main__":
    print("Testing HVAC Environment\n")
    
    env = HVACEnvironment()
    baseline = BaselineHVACController()
    
    start_time = datetime(2024, 1, 1, 0, 0)
    steps = int(24 * 60 / TIME_STEP_MINUTES)
    
    print("Running 24-hour simulation...")
    
    for step in range(steps):
        current_time = start_time + timedelta(minutes=step * TIME_STEP_MINUTES)
        state = env.get_current_state()
        hvac_action, pcm_action = baseline.decide(state['indoor_temp'])
        new_state = env.step(current_time, hvac_action, pcm_action)
        
        if step % 16 == 0:
            print(f"{current_time.strftime('%H:%M')} - "
                  f"Indoor: {new_state['indoor_temp']:.1f}°C, "
                  f"Occupancy: {new_state['occupancy']}, "
                  f"HVAC: {hvac_action}")
    
    print(f"\n--- Results ---")
    print(f"Total HVAC Energy: {env.total_hvac_energy:.2f} kWh")
    print(f"Total Cost: ₹{env.get_total_cost():.2f}")