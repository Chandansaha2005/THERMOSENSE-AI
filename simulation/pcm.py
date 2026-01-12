"""
Phase Change Material (PCM) Thermal Storage Model
Simulates thermal battery behavior
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *


class PCMStorage:
    """Phase Change Material thermal energy storage system"""
    
    def __init__(
        self,
        max_capacity=PCM_MAX_CAPACITY,
        charge_rate=PCM_CHARGE_RATE,
        discharge_rate=PCM_DISCHARGE_RATE,
        efficiency=PCM_EFFICIENCY,
        initial_charge=PCM_INITIAL_CHARGE
    ):
        """Initialize PCM storage"""
        self.max_capacity = max_capacity
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
        self.efficiency = efficiency
        self.state_of_charge = initial_charge
        
        self.total_energy_charged = 0.0
        self.total_energy_discharged = 0.0
        self.cycles = 0
        
    def charge(self, time_step_hours):
        """Charge PCM with cooling energy"""
        max_energy = self.charge_rate * time_step_hours
        available_capacity = self.max_capacity - self.state_of_charge
        energy_charged = min(max_energy, available_capacity)
        
        self.state_of_charge += energy_charged * self.efficiency
        self.total_energy_charged += energy_charged
        
        return energy_charged
    
    def discharge(self, time_step_hours, requested_cooling_kw=None):
        """Discharge PCM to provide cooling"""
        if requested_cooling_kw is not None:
            discharge_kw = min(requested_cooling_kw, self.discharge_rate)
        else:
            discharge_kw = self.discharge_rate
        
        max_energy = discharge_kw * time_step_hours
        available_energy = self.state_of_charge
        energy_discharged = min(max_energy, available_energy)
        
        self.state_of_charge -= energy_discharged
        self.total_energy_discharged += energy_discharged
        
        actual_cooling_kw = energy_discharged / time_step_hours if time_step_hours > 0 else 0.0
        
        return actual_cooling_kw
    
    def get_state_of_charge(self):
        """Return current charge (kWh)"""
        return self.state_of_charge
    
    def get_state_of_charge_percent(self):
        """Return charge as percentage"""
        return (self.state_of_charge / self.max_capacity) * 100
    
    def can_charge(self):
        """Check if PCM can accept charge"""
        return self.state_of_charge < (self.max_capacity * PCM_MAX_CHARGE_THRESHOLD)
    
    def can_discharge(self):
        """Check if PCM has useful charge"""
        return self.state_of_charge > (self.max_capacity * PCM_MIN_USEFUL_CHARGE)
    
    def get_available_cooling_capacity(self):
        """Return available cooling (kWh)"""
        return self.state_of_charge
    
    def get_available_charging_capacity(self):
        """Return available space (kWh)"""
        return self.max_capacity - self.state_of_charge
    
    def reset(self):
        """Reset to initial state"""
        self.state_of_charge = PCM_INITIAL_CHARGE
        self.total_energy_charged = 0.0
        self.total_energy_discharged = 0.0
        self.cycles = 0
    
    def get_stats(self):
        """Return statistics"""
        return {
            'state_of_charge_kwh': self.state_of_charge,
            'state_of_charge_percent': self.get_state_of_charge_percent(),
            'total_charged_kwh': self.total_energy_charged,
            'total_discharged_kwh': self.total_energy_discharged,
            'efficiency': self.efficiency,
            'capacity_kwh': self.max_capacity
        }


if __name__ == "__main__":
    print("Testing PCM Storage Model\n")
    
    pcm = PCMStorage()
    time_step = TIME_STEP_MINUTES / 60.0
    
    print(f"Initial State: {pcm.get_state_of_charge_percent():.1f}%")
    
    print("\n--- Charging Phase ---")
    for i in range(4):
        energy = pcm.charge(time_step)
        print(f"Step {i+1}: Charged {energy:.2f} kWh, "
              f"SoC: {pcm.get_state_of_charge_percent():.1f}%")
    
    print("\n--- Discharging Phase ---")
    for i in range(4):
        cooling_kw = pcm.discharge(time_step, requested_cooling_kw=2.0)
        print(f"Step {i+1}: Provided {cooling_kw:.2f} kW cooling, "
              f"SoC: {pcm.get_state_of_charge_percent():.1f}%")
    
    print("\n--- Final Stats ---")
    stats = pcm.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")