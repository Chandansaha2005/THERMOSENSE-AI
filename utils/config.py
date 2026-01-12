"""
Configuration file for THERMOSENSE-AI system
Contains all system parameters and constants
"""

# Simulation Parameters
SIMULATION_HOURS = 168  # 1 week
TIME_STEP_MINUTES = 15  # Data point every 15 minutes
ROOM_VOLUME = 50  # m³
ROOM_SURFACE_AREA = 75  # m²

# Temperature Parameters
COMFORT_MIN = 24.0  # °C
COMFORT_MAX = 26.0  # °C
OUTDOOR_TEMP_MIN = 28.0  # °C
OUTDOOR_TEMP_MAX = 38.0  # °C
INITIAL_ROOM_TEMP = 28.0  # °C

# HVAC Parameters
HVAC_COOLING_CAPACITY = 3.5  # kW
HVAC_COP = 3.0  # Coefficient of Performance
HVAC_RESPONSE_TIME = 0.25  # hours

# PCM Parameters
PCM_MAX_CAPACITY = 15.0  # kWh of cooling energy
PCM_CHARGE_RATE = 2.5  # kW
PCM_DISCHARGE_RATE = 2.0  # kW
PCM_EFFICIENCY = 0.85  # 85% round-trip efficiency
PCM_INITIAL_CHARGE = 0.0  # Start empty

# Occupancy Parameters
MAX_OCCUPANCY = 10
HEAT_PER_PERSON = 100  # Watts per person
OFFICE_START_HOUR = 9
OFFICE_END_HOUR = 17

# Energy Cost Parameters (₹/kWh)
PEAK_ELECTRICITY_COST = 8.5  # 2 PM - 6 PM
STANDARD_ELECTRICITY_COST = 6.0  # 6 AM - 10 PM
OFF_PEAK_ELECTRICITY_COST = 3.0  # 10 PM - 6 AM

# ML Model Parameters
PREDICTION_HORIZON = 30  # minutes
OCCUPANCY_MODEL_PATH = "models/occupancy_model.pkl"
TEMPERATURE_MODEL_PATH = "models/temperature_model.pkl"

# Feature Engineering
LAG_FEATURES = [15, 30, 60]  # minutes

# Thermal Physics Constants
AIR_SPECIFIC_HEAT = 1005  # J/(kg·K)
AIR_DENSITY = 1.2  # kg/m³
HEAT_TRANSFER_COEFF = 5.0  # W/(m²·K) - simplified U-value
SOLAR_GAIN_FACTOR = 0.3  # kW during daytime

# Controller Parameters
DECISION_INTERVAL = 15  # minutes
TEMPERATURE_DEADBAND = 0.5  # °C
PCM_MIN_USEFUL_CHARGE = 0.2  # 20% minimum to use PCM
PCM_MAX_CHARGE_THRESHOLD = 0.9  # 90% max to avoid overcharging

# Data Paths
DATA_DIR = "data"
MODEL_DIR = "models"
SIMULATED_DATA_PATH = "data/simulated_data.csv"
RESULTS_PATH = "data/results.csv"