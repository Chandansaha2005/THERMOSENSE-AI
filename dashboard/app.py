"""
THERMOSENSE-AI Dashboard
Interactive visualization and simulation interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import *
from simulation.environment import HVACEnvironment, BaselineHVACController
from ml.occupancy_model import OccupancyPredictor
from ml.heat_model import TemperaturePredictor
from controller.thermal_controller import ThermalBudgetController


st.set_page_config(
    page_title="THERMOSENSE-AI",
    page_icon="🌡️",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained ML models"""
    occ_pred = OccupancyPredictor()
    temp_pred = TemperaturePredictor()
    
    try:
        occ_pred.load()
        temp_pred.load()
        return occ_pred, temp_pred, True
    except:
        return None, None, False


def run_simulation(hours, use_ai=True):
    """Run HVAC simulation"""
    env = HVACEnvironment()
    
    if use_ai:
        occ_pred, temp_pred, loaded = load_models()
        if not loaded:
            st.error("Models not loaded. Using baseline.")
            controller = BaselineHVACController()
        else:
            controller = ThermalBudgetController(occ_pred, temp_pred)
    else:
        controller = BaselineHVACController()
    
    start_time = datetime(2024, 1, 1, 0, 0)
    steps = int(hours * 60 / TIME_STEP_MINUTES)
    history = []
    
    for step in range(steps):
        current_time = start_time + timedelta(minutes=step * TIME_STEP_MINUTES)
        current_env_state = env.get_current_state()
        
        state = {
            'indoor_temp': current_env_state['indoor_temp'],
            'outdoor_temp': env.outdoor_temp_sensor.read(current_time.hour, current_time.weekday()),
            'occupancy': env.occupancy_sensor.read(current_time.hour, current_time.weekday()),
            'pcm_charge_kwh': current_env_state['pcm_charge_kwh'],
            'pcm_charge_percent': current_env_state['pcm_charge_percent'],
            'hour': current_time.hour,
            'day_of_week': current_time.weekday(),
            'timestamp': current_time
        }
        
        if len(history) > 0:
            state.update(history[-1])
        
        if use_ai and isinstance(controller, ThermalBudgetController):
            hvac_action, pcm_action, reason = controller.decide(state)
        else:
            hvac_action, pcm_action = controller.decide(state['indoor_temp'])
        
        new_state = env.step(current_time, hvac_action, pcm_action)
        history.append(new_state)
    
    return env, env.get_history_df()


def create_temperature_plot(df, baseline_df=None):
    """Temperature comparison plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['indoor_temp'],
        name='THERMOSENSE-AI',
        line=dict(color='#1f77b4', width=2)
    ))
    
    if baseline_df is not None:
        fig.add_trace(go.Scatter(
            x=baseline_df['timestamp'], y=baseline_df['indoor_temp'],
            name='Baseline HVAC',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
    
    fig.add_hrect(
        y0=COMFORT_MIN, y1=COMFORT_MAX,
        fillcolor="green", opacity=0.1,
        annotation_text="Comfort Zone"
    )
    
    fig.update_layout(
        title="Indoor Temperature",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        height=400
    )
    
    return fig


def create_pcm_plot(df):
    """PCM charge level plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['pcm_charge_percent'],
        fill='tozeroy',
        name='PCM Charge',
        line=dict(color='#2ca02c', width=2)
    ))
    
    fig.update_layout(
        title="PCM Storage Level",
        xaxis_title="Time",
        yaxis_title="Charge (%)",
        height=350
    )
    
    return fig


def create_action_timeline(df):
    """Action timeline"""
    df = df.copy()
    df['action_code'] = 0
    df.loc[df['hvac_action'] == 'ON', 'action_code'] = 3
    df.loc[df['pcm_action'] == 'CHARGE', 'action_code'] = 1
    df.loc[df['pcm_action'] == 'DISCHARGE', 'action_code'] = 2
    
    colors = {0: 'gray', 1: '#2ca02c', 2: '#1f77b4', 3: '#d62728'}
    labels = {0: 'Coast', 1: 'Charge PCM', 2: 'Discharge PCM', 3: 'Direct HVAC'}
    
    fig = go.Figure()
    
    for code, label in labels.items():
        mask = df['action_code'] == code
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df.loc[mask, 'timestamp'],
                y=df.loc[mask, 'action_code'],
                mode='markers',
                name=label,
                marker=dict(size=8, color=colors[code])
            ))
    
    fig.update_layout(
        title="Control Actions",
        xaxis_title="Time",
        yaxis=dict(
            tickvals=[0, 1, 2, 3],
            ticktext=['Coast', 'Charge', 'Discharge', 'HVAC']
        ),
        height=300
    )
    
    return fig


def create_comparison(ai_df, baseline_df):
    """Energy comparison"""
    time_step_h = TIME_STEP_MINUTES / 60.0
    
    ai_energy = (ai_df['hvac_state'] * HVAC_COOLING_CAPACITY / HVAC_COP * time_step_h).sum()
    baseline_energy = (baseline_df['hvac_state'] * HVAC_COOLING_CAPACITY / HVAC_COP * time_step_h).sum()
    
    ai_df['hvac_energy'] = ai_df['hvac_state'] * (HVAC_COOLING_CAPACITY / HVAC_COP) * time_step_h
    ai_df['cost'] = ai_df['hvac_energy'] * ai_df['electricity_cost_per_kwh']
    ai_cost = ai_df['cost'].sum()
    
    baseline_df['hvac_energy'] = baseline_df['hvac_state'] * (HVAC_COOLING_CAPACITY / HVAC_COP) * time_step_h
    baseline_df['cost'] = baseline_df['hvac_energy'] * baseline_df['electricity_cost_per_kwh']
    baseline_cost = baseline_df['cost'].sum()
    
    return {
        'ai_energy': ai_energy,
        'baseline_energy': baseline_energy,
        'energy_savings': baseline_energy - ai_energy,
        'energy_reduction_pct': ((baseline_energy - ai_energy) / baseline_energy) * 100,
        'ai_cost': ai_cost,
        'baseline_cost': baseline_cost,
        'cost_savings': baseline_cost - ai_cost,
        'cost_reduction_pct': ((baseline_cost - ai_cost) / baseline_cost) * 100
    }


def main():
    st.markdown('<p class="main-header">🌡️ THERMOSENSE-AI</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-Driven Thermal Budgeting for Energy-Efficient HVAC</p>', unsafe_allow_html=True)
    
    st.sidebar.title("⚙️ Settings")
    
    hours = st.sidebar.slider(
        "Simulation Duration (hours)",
        24, 168, 72, 24
    )
    
    compare = st.sidebar.checkbox("Compare with Baseline", value=True)
    
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            ai_env, ai_df = run_simulation(hours, use_ai=True)
            st.session_state['ai_df'] = ai_df
            st.session_state['ai_env'] = ai_env
            
            if compare:
                baseline_env, baseline_df = run_simulation(hours, use_ai=False)
                st.session_state['baseline_df'] = baseline_df
                st.session_state['baseline_env'] = baseline_env
            
            st.success("✅ Simulation complete!")
    
    if 'ai_df' in st.session_state:
        ai_df = st.session_state['ai_df']
        baseline_df = st.session_state.get('baseline_df', None)
        
        st.markdown("### 📊 Key Metrics")
        
        if baseline_df is not None:
            comp = create_comparison(ai_df, baseline_df)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Energy Savings",
                    f"{comp['energy_savings']:.1f} kWh",
                    f"-{comp['energy_reduction_pct']:.1f}%"
                )
            
            with col2:
                st.metric(
                    "Cost Savings",
                    f"₹{comp['cost_savings']:.2f}",
                    f"-{comp['cost_reduction_pct']:.1f}%"
                )
            
            with col3:
                st.metric("AI System Cost", f"₹{comp['ai_cost']:.2f}")
            
            with col4:
                st.metric("Baseline Cost", f"₹{comp['baseline_cost']:.2f}")
        
        st.markdown("---")
        
        st.plotly_chart(create_temperature_plot(ai_df, baseline_df), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_pcm_plot(ai_df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_action_timeline(ai_df), use_container_width=True)
        
        with st.expander("📋 View Data"):
            st.dataframe(ai_df[['timestamp', 'indoor_temp', 'occupancy', 
                                'hvac_action', 'pcm_action', 'pcm_charge_percent']].head(50))
    
    else:
        st.info("👈 Configure settings and click 'Run Simulation'")
        
        st.markdown("### 🏗️ System Architecture")
        st.markdown("""
        **THERMOSENSE-AI** uses thermal budgeting to optimize HVAC:
        
        1. **IoT Sensors** - Temperature, occupancy, environment data
        2. **ML Predictors** - Forecast occupancy and heat load 30 min ahead
        3. **Thermal Budget Controller** - Decides when and how to cool
        4. **PCM Storage** - Thermal battery for cooling energy
        
        **Innovation:** Treats cooling as a manageable resource, storing when cheap, 
        using when expensive.
        """)


if __name__ == "__main__":
    main()