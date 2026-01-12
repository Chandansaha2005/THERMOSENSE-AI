"""
Thermal Budget Controller - CORE INNOVATION
AI-powered decision engine for HVAC and PCM control
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *


class ThermalBudgetController:
    """
    Intelligent thermal budget controller
    
    Makes decisions on HVAC and PCM based on:
    - ML predictions of occupancy and temperature
    - Electricity costs
    - PCM state of charge
    - Comfort constraints
    """
    
    def __init__(self, occupancy_predictor, temperature_predictor):
        """Initialize with trained ML models"""
        self.occupancy_predictor = occupancy_predictor
        self.temperature_predictor = temperature_predictor
        self.decisions = []
        
    def get_electricity_cost(self, hour):
        """Get electricity cost for hour"""
        if 14 <= hour < 18:
            return PEAK_ELECTRICITY_COST
        elif 6 <= hour < 22:
            return STANDARD_ELECTRICITY_COST
        else:
            return OFF_PEAK_ELECTRICITY_COST
    
    def predict_future_state(self, current_state):
        """Predict future occupancy and temperature"""
        try:
            predicted_occupancy = self.occupancy_predictor.predict(current_state)
            predicted_temp = self.temperature_predictor.predict(current_state)
        except:
            predicted_occupancy = current_state.get('occupancy', 0)
            predicted_temp = current_state.get('indoor_temp', INITIAL_ROOM_TEMP)
        
        return {
            'predicted_occupancy': predicted_occupancy,
            'predicted_temp': predicted_temp
        }
    
    def calculate_thermal_cost_score(self, state, predictions, electricity_cost):
        """
        Calculate thermal cost score
        
        Higher score = more expensive to cool now
        """
        occupancy_heat_kw = (predictions['predicted_occupancy'] * HEAT_PER_PERSON) / 1000.0
        temp_trend = predictions['predicted_temp'] - state['indoor_temp']
        thermal_load_kw = occupancy_heat_kw + max(0, temp_trend * 2)
        
        hvac_cost = thermal_load_kw * (electricity_cost / HVAC_COP)
        
        pcm_charge_percent = state.get('pcm_charge_percent', 0)
        pcm_benefit = (pcm_charge_percent / 100) * PCM_DISCHARGE_RATE * 0.5
        
        thermal_cost_score = hvac_cost - pcm_benefit
        
        return thermal_cost_score
    
    def decide(self, state):
        """
        Make HVAC and PCM control decision
        
        Returns: (hvac_action, pcm_action, reason)
        """
        indoor_temp = state['indoor_temp']
        hour = state['hour']
        pcm_charge_percent = state.get('pcm_charge_percent', 0)
        
        electricity_cost = self.get_electricity_cost(hour)
        predictions = self.predict_future_state(state)
        thermal_cost_score = self.calculate_thermal_cost_score(
            state, predictions, electricity_cost
        )
        
        # DECISION LOGIC
        
        # Priority 1: Comfort violation - immediate action
        if indoor_temp > COMFORT_MAX:
            if pcm_charge_percent > PCM_MIN_USEFUL_CHARGE * 100 and thermal_cost_score > 3.0:
                decision = ('OFF', 'DISCHARGE', 
                           f'Comfort violation - Using PCM (score: {thermal_cost_score:.1f})')
            else:
                decision = ('ON', 'IDLE', 
                           'Comfort violation - Direct HVAC')
        
        # Priority 2: Predicted comfort violation
        elif predictions['predicted_temp'] > COMFORT_MAX:
            if pcm_charge_percent > PCM_MIN_USEFUL_CHARGE * 100:
                decision = ('OFF', 'DISCHARGE', 
                           'Predicted overheat - Pre-cooling with PCM')
            else:
                decision = ('ON', 'IDLE', 
                           'Predicted overheat - Pre-cooling with HVAC')
        
        # Priority 3: Approaching upper limit
        elif indoor_temp > (COMFORT_MAX - 0.5):
            if pcm_charge_percent > PCM_MIN_USEFUL_CHARGE * 100 and electricity_cost > STANDARD_ELECTRICITY_COST:
                decision = ('OFF', 'DISCHARGE', 
                           f'High cost (₹{electricity_cost}/kWh) - Using PCM')
            else:
                decision = ('ON', 'IDLE', 
                           'Approaching limit - HVAC cooling')
        
        # Priority 4: Good conditions for PCM charging
        elif (electricity_cost < STANDARD_ELECTRICITY_COST and 
              predictions['predicted_occupancy'] < 3 and
              pcm_charge_percent < PCM_MAX_CHARGE_THRESHOLD * 100 and
              indoor_temp < COMFORT_MIN + 0.5):
            decision = ('OFF', 'CHARGE', 
                       f'Low cost (₹{electricity_cost}/kWh) - Charging PCM')
        
        # Priority 5: Below comfort minimum
        elif indoor_temp < COMFORT_MIN:
            decision = ('OFF', 'IDLE', 
                       'Below minimum - Coast')
        
        # Priority 6: Comfortable
        else:
            decision = ('OFF', 'IDLE', 
                       f'Comfortable ({indoor_temp:.1f}°C) - Coast')
        
        # Record decision
        decision_record = {
            'timestamp': state.get('timestamp'),
            'indoor_temp': indoor_temp,
            'predicted_temp': predictions['predicted_temp'],
            'predicted_occupancy': predictions['predicted_occupancy'],
            'pcm_charge_percent': pcm_charge_percent,
            'electricity_cost': electricity_cost,
            'thermal_cost_score': thermal_cost_score,
            'hvac_action': decision[0],
            'pcm_action': decision[1],
            'reason': decision[2]
        }
        self.decisions.append(decision_record)
        
        return decision
    
    def get_decision_history(self):
        """Return decision history"""
        import pandas as pd
        return pd.DataFrame(self.decisions)
    
    def reset_history(self):
        """Clear history"""
        self.decisions = []


if __name__ == "__main__":
    print("=== Thermal Budget Controller Test ===\n")
    
    from ml.occupancy_model import OccupancyPredictor
    from ml.heat_model import TemperaturePredictor
    
    if not os.path.exists(OCCUPANCY_MODEL_PATH) or not os.path.exists(TEMPERATURE_MODEL_PATH):
        print("Error: Models not found")
        print("Run: python ml/occupancy_model.py")
        print("Run: python ml/heat_model.py")
        exit(1)
    
    print("Loading models...")
    occ_pred = OccupancyPredictor()
    occ_pred.load()
    
    temp_pred = TemperaturePredictor()
    temp_pred.load()
    
    controller = ThermalBudgetController(occ_pred, temp_pred)
    print("✓ Controller initialized\n")
    
    scenarios = [
        {
            'name': 'High temp, peak hour',
            'state': {
                'indoor_temp': 26.5,
                'outdoor_temp': 35.0,
                'occupancy': 8,
                'pcm_charge_percent': 70,
                'hour': 15,
                'timestamp': '2024-01-01 15:00'
            }
        },
        {
            'name': 'Night, low cost',
            'state': {
                'indoor_temp': 24.5,
                'outdoor_temp': 28.0,
                'occupancy': 0,
                'pcm_charge_percent': 30,
                'hour': 2,
                'timestamp': '2024-01-01 02:00'
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"--- {scenario['name']} ---")
        hvac, pcm, reason = controller.decide(scenario['state'])
        print(f"HVAC: {hvac}, PCM: {pcm}")
        print(f"Reason: {reason}\n")
    
    print("✓ Controller test complete")