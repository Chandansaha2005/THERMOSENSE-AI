# 🌡️ THERMOSENSE-AI

## AI-Driven Thermal Budgeting System for Predictive, Energy-Efficient HVAC Control

<p align="center">
  <a href="https://thermosense-ai.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/THERMOSENSE%20AI-Click%20Here-00C853?style=for-the-badge&logo=google-chrome&logoColor=white" alt="THERMOSENSE-AI Demo"/>
  </a>
</p>

## 🎯 Problem Statement

HVAC systems consume **40% of building energy** but operate using primitive reactive control:

### Current Limitations
- ❌ **Reactive** - Cooling starts ONLY after rooms become hot
- ❌ **No Prediction** - Cannot anticipate occupancy or heat loads
- ❌ **Energy Waste** - Run during expensive peak hours
- ❌ **Peak Demand** - Stress electrical grids
- ❌ **No Intelligence** - Cannot learn or optimize

**Result:** Buildings waste 30-40% of cooling energy and pay premium electricity prices.

---

## 💡 Our Solution: Thermal Budgeting

**THERMOSENSE-AI** introduces a revolutionary approach: treat cooling as a **manageable budget**, not an unlimited reaction.

### The Core Concept

> **Traditional HVAC asks: "Is the room hot?"**  
> **THERMOSENSE-AI asks: "When is the best time and method to cool?"**

The system:
- 🔮 **Predicts** occupancy and heat loads 30 minutes ahead
- 🧠 **Decides** when to cool, how to cool, and whether to store cooling
- 💰 **Optimizes** for comfort AND cost
- ⚡ **Integrates** Phase Change Materials as thermal batteries

---

## 🔥 Key Innovation: Thermal Budget Controller

### Decision Logic

At every time step, the controller evaluates:

**Inputs:**
- Current & predicted temperature
- Current & predicted occupancy  
- PCM storage level (0-100%)
- Electricity price (₹/kWh)
- Comfort constraints (24-26°C)

**Outputs:**
- HVAC: ON or OFF
- PCM: CHARGE, DISCHARGE, or IDLE

**Example:**
```
2:00 AM - Low electricity (₹3/kWh), no occupancy
→ Decision: CHARGE PCM (store cooling)

2:00 PM - High electricity (₹8.5/kWh), meeting predicted
→ Decision: DISCHARGE PCM (use stored cooling, save ₹5.50)
```

---

## 🏗️ System Architecture


<p align="center">
  <img src="assets/WorkflowDiagram.png" alt="THERMOSENSE-AI Workflow" width="500px" style="margin: 20px 0; border-radius: 10px;"/>
</p>

**End-to-End Pipeline:**

```
IoT Sensors → Feature Engineering → ML Models → Thermal Controller
     ↓              ↓                   ↓              ↓
Temperature    Lag Features      Occupancy        HVAC Actions
Occupancy      Time Encoding     Temperature      PCM Control
Electricity    Rolling Stats     Predictions      Dashboard
Rates                                             Visualization
```

### Core Components

| Component | Purpose | Details |
|-----------|---------|---------|
| **Simulated IoT** | Data Collection | Temperature, Occupancy, HVAC state with realistic patterns |
| **Feature Engineering** | Data Preparation | 13-19 intelligent features (lags, cyclic encoding, rolling stats) |
| **ML Models** | Prediction | Occupancy (RandomForest 85-90%), Temperature (GradientBoosting 90-95%) |
| **Thermal Controller** | Decision Engine | Cost-optimized decisions with comfort priority |
| **PCM Storage** | Energy Buffer | 15 kWh thermal battery (85% efficiency) |
| **Dashboard** | Visualization | Real-time monitoring via Streamlit |

---

## 🚀 Installation & Quick Start

### ⚡ Quick Start (3 Commands)

```bash
# 1️⃣ Install dependencies
pip install -r requirements.txt

# 2️⃣ Generate data & train models (25 seconds)
python simulation/sensors.py
python ml/occupancy_model.py
python ml/heat_model.py

# 3️⃣ Launch interactive dashboard
streamlit run dashboard/app.py
```

**Dashboard opens at:** 🔗 `http://localhost:8501`

### 📋 Detailed Setup

| Step | Command | Duration | Output |
|:----:|:-------:|:--------:|:------:|
| Install | `pip install -r requirements.txt` | 1 min | All dependencies |
| Generate Data | `python simulation/sensors.py` | 30 sec | `data/simulated_data.csv` |
| Train Occupancy | `python ml/occupancy_model.py` | 10 sec | `models/occupancy_model.pkl` |
| Train Temperature | `python ml/heat_model.py` | 15 sec | `models/temperature_model.pkl` |
| **Run Dashboard** | `streamlit run dashboard/app.py` | Instant | Interactive UI |

**Total Setup Time:** ~2 minutes ⏱️

### 🔧 Advanced Configuration

Edit `utils/config.py` to customize:

```python
# Modify these to experiment
COMFORT_RANGE = (24, 26)        # Temperature setpoint (°C)
PCM_CAPACITY = 15               # Thermal storage (kWh)
HVAC_POWER = 3.5                # HVAC capacity (kW)
PREDICTION_HORIZON = 30         # Forecast window (minutes)
PEAK_HOUR_START = 14            # Peak tariff start (2 PM)
```

---

## 📁 Project Structure

```
thermosense-ai/
│
├── 📊 dashboard/
│   └── app.py                          # Streamlit interactive dashboard
│
├── 🤖 ml/
│   ├── features.py                     # Feature engineering pipeline
│   ├── occupancy_model.py              # Occupancy predictor (RandomForest)
│   └── heat_model.py                   # Temperature predictor (GradientBoosting)
│
├── 🏗️ simulation/
│   ├── sensors.py                      # IoT sensor simulation
│   ├── pcm.py                          # Phase Change Material model
│   └── environment.py                  # HVAC environment dynamics
│
├── 🎛️ controller/
│   └── thermal_controller.py           # Decision engine (Core Innovation)
│
├── 💾 data/
│   └── simulated_data.csv              # Generated training data
│
├── 🎯 models/
│   ├── occupancy_model.pkl             # Trained occupancy model
│   └── temperature_model.pkl           # Trained temperature model
│
├── ⚙️ utils/
│   └── config.py                       # Configuration & parameters
│
├── 📦 requirements.txt                 # Python dependencies
├── 📖 README.md                        # This file
└── 📜 LICENSE                          # MIT License
```

---

## 🎯 How It Works

### 1️⃣ Data Collection
Realistic simulation of building sensors with daily/weekly patterns

### 2️⃣ Feature Engineering
Intelligent preprocessing: lag features, time encodings, rolling statistics

### 3️⃣ ML Predictions
- **Occupancy** 30 min ahead → RandomForest classifier
- **Temperature** 30 min ahead → GradientBoosting regressor

### 4️⃣ Thermal Budget Controller ⭐

The **decision engine** evaluates real-time and predicted conditions:

**Thermal Cost Score:**
```
Score = (Predicted Load × Electricity Price) - (PCM Benefit)
```

**Decision Priority:**
1. ❌ **Comfort Violation** → Immediate HVAC action
2. ⚠️ **Predicted Violation** → Pre-cooling strategy
3. 💰 **High Cost + PCM Available** → Discharge thermal storage
4. 🔋 **Low Cost + Low Occupancy** → Charge PCM (store energy)
5. ✅ **Comfortable & Optimized** → Coast (no action)

### 5️⃣ PCM Thermal Storage
Simulates phase change material battery:
- **Charges** at night (₹3/kWh) → Store cooling for peak hours
- **Discharges** during peak (₹8.5/kWh) → Use stored energy, save money
- 85% round-trip efficiency

### 6️⃣ Dashboard Visualization
<p align="center">
  <img src="assets/temp&pcm.png" alt="Temperature & PCM Dashboard" width="5000px" style="margin: 20px 0; border-radius: 8px;"/>
</p>

<p align="center">
  <img src="assets/data.png" alt="System Data Flow" width="5000px" style="margin: 20px 0; border-radius: 8px;"/>
</p>

Real-time monitoring of:
- Temperature trends & comfort zones
- PCM charge/discharge cycles
- Action timeline (HVAC ON/OFF, PCM state)
- Energy consumption & cost savings

---

## 📊 Performance Metrics

### Typical 72-Hour Simulation Results

<p align="center">
  <img src="assets/keymetrics.png" alt="Key Performance Metrics" width="700px" style="margin: 20px 0; border-radius: 8px;"/>
</p>

| Metric | Baseline | THERMOSENSE-AI | Improvement |
|:------:|:--------:|:--------------:|:-----------:|
| **Energy Usage** | 89.5 kWh | 62.1 kWh | **-30.6%** ⚡ |
| **Cost** | ₹537.50 | ₹356.20 | **-33.7%** 💰 |
| **Peak Demand** | 7.2 kW | 4.8 kW | **-33.3%** 📉 |
| **Comfort Score** | 100% | 100% | **Same** ✅ |

### 🎯 Key Outcomes

✅ **27-35% energy reduction** through intelligent scheduling  
✅ **30-40% cost savings** by shifting loads to off-peak hours  
✅ **30-35% peak demand reduction** easing grid stress  
✅ **0% comfort violations** with comfort-first logic  
✅ **Shifts consumption** from peak (₹8.5/kWh) to off-peak (₹3/kWh)  

### 💼 Real-World ROI (500 m² Office)

| Metric | Annual Impact |
|:-------|:-------------:|
| **Energy Savings** | ~12,000 kWh 🔌 |
| **Cost Savings** | ₹72,000 (~$900) 💵 |
| **CO₂ Reduction** | ~10 tons 🌱 |
| **Payback Period** | 8-12 months 📅 |

---

## 🔧 Technical Specifications

### 🤖 Machine Learning Models

| Model | Purpose | Architecture | Accuracy |
|:-----:|:-------:|:------------:|:--------:|
| **RandomForest** | Occupancy Prediction | 100 trees, depth 15 | 85-90% |
| **GradientBoosting** | Temperature Prediction | 150 trees, LR 0.1 | 90-95% |

**Training:** < 30 seconds total | **Features:** 13-19 intelligent features

### ⚙️ System Parameters

```python
# Comfort & HVAC
COMFORT_RANGE        = 24-26°C (72.2-78.8°F)
HVAC_CAPACITY        = 3.5 kW
HVAC_COP             = 3.0 (Coefficient of Performance)

# Thermal Storage (PCM Battery)
PCM_CAPACITY         = 15 kWh
PCM_CHARGE_RATE      = 2.5 kW
PCM_DISCHARGE_RATE   = 2.0 kW
PCM_EFFICIENCY       = 85%

# Prediction & Control
PREDICTION_HORIZON   = 30 minutes
CONTROL_INTERVAL     = 15 minutes
DECISION_UPDATE_RATE = Every 15 minutes
```

### 💷 Electricity Tariff (Indian Model)

```
Peak Hours (2-6 PM):     ₹8.5/kWh  🔴
Standard (6 AM-10 PM):   ₹6.0/kWh  🟡
Off-Peak (10 PM-6 AM):   ₹3.0/kWh  🟢
```

**Smart Strategy:** Pre-cool at night (₹3) → Use stored cooling during peak (₹8.5) = **₹5.50/kWh savings**

---

## 🏆 Why THERMOSENSE-AI?

### ✨ Technical Excellence
- ✅ **Complete working system** - Not mockups or slides
- ✅ **Sophisticated ML pipeline** - Professional-grade models
- ✅ **Novel approach** - Thermal budgeting is unique in HVAC
- ✅ **Production-ready code** - Fully documented, optimized

### 💼 Business Viability
- ✅ **Clear ROI** - 10-month payback period
- ✅ **Scalable** - Works for homes, offices, hospitals, data centers
- ✅ **Retrofit-compatible** - No major infrastructure changes
- ✅ **Solves major problem** - HVAC is 40% of building energy

### 🌍 Impact
- ✅ **Reduces CO₂** - ~10 tons/year per building
- ✅ **Eases grid stress** - Flattens peak demand by 33%
- ✅ **Saves money** - ₹72,000/year per 500m² office
- ✅ **Improves comfort** - Zero thermal violations  

---

## ❓ FAQ

<details>
<summary><strong>Q: Is this just a simulation?</strong></summary>

A: Yes, for this hackathon version. The physics models and ML pipeline are production-ready. Phase 2 will integrate real IoT sensors and HVAC hardware.

</details>

<details>
<summary><strong>Q: How long does it take to train models?</strong></summary>

A: Only **25 seconds total** - RandomForest (10s) + GradientBoosting (15s). Lightweight and edge-deployable. ⚡

</details>

<details>
<summary><strong>Q: What if ML predictions are wrong?</strong></summary>

A: **Comfort is the priority.** If temperature exceeds 26°C, the system immediately activates HVAC regardless of predictions. Real-time sensor data always overrides forecasts.

</details>

<details>
<summary><strong>Q: Can I customize the system?</strong></summary>

A: Absolutely! Edit `utils/config.py` to change comfort range, PCM size, HVAC power, electricity tariffs, etc. See Advanced Configuration section above.

</details>

<details>
<summary><strong>Q: What's the prediction accuracy?</strong></summary>

A: 
- **Occupancy:** 85-90% (Mean Absolute Error: 1.5 persons)
- **Temperature:** 90-95% (MAE: 0.5°C)

Trained on realistic patterns with noise.

</details>

<details>
<summary><strong>Q: Can this work in other countries?</strong></summary>

A: Yes! The system is tariff-agnostic. Modify electricity prices in `config.py` for your region (USA: $/kWh, Europe: €/kWh, etc.).

</details>

---

## 🚨 Troubleshooting Guide

| Issue | Solution | Code |
|:-----:|:--------:|:----:|
| ❌ Import Error | Install all dependencies | `pip install -r requirements.txt` |
| ❌ Data Missing | Generate sensor simulation | `python simulation/sensors.py` |
| ❌ Models Missing | Train ML models | `python ml/occupancy_model.py && python ml/heat_model.py` |
| ❌ Dashboard Won't Start | Try alternate port | `streamlit run dashboard/app.py --server.port 8502` |
| ❌ Permission Denied | Use sudo (Linux/Mac) | `sudo python ...` |
| ❌ Out of Memory | Reduce data generation | Edit `simulation/sensors.py` parameter `n_days=1` |

**Need more help?** Check the inline code comments - every function is fully documented! 📝

---

## 🔮 Future Roadmap

### 🔜 Phase 2 Enhancements

- 🔌 **Real Hardware Integration** - Connect actual IoT sensors & smart meters
- 🏢 **Multi-Room Systems** - Extend to entire buildings with zone-based control
- 🧠 **Deep Learning** - LSTM networks for longer-term predictions
- ☀️ **Renewable Integration** - Prioritize solar energy during generation peaks
- 🌍 **Cloud Dashboard** - Remote monitoring across multiple properties
- 📱 **Mobile App** - Real-time notifications & manual overrides
- 🔗 **API Interface** - Integration with building management systems (BMS)
- 🤝 **Multi-Building** - District-level energy optimization

### 🎯 Success Metrics

- ✅ Deploy in 5+ real buildings
- ✅ Achieve 35-40% verified energy savings
- ✅ Hit 12-month payback period
- ✅ Generate 1000+ tons CO₂ reduction across fleet

---

## � Learning Resources

### Understanding the System

| Topic | Resource | Learning Time |
|:-----:|:--------:|:-------------:|
| **Thermal Physics** | `simulation/environment.py` | 10 min |
| **PCM Storage** | `simulation/pcm.py` | 8 min |
| **Feature Engineering** | `ml/features.py` | 12 min |
| **ML Models** | `ml/occupancy_model.py`, `ml/heat_model.py` | 15 min |
| **Decision Logic** | `controller/thermal_controller.py` | 15 min |

### Key References

- 📖 **PCM Technology:** NREL Thermal Energy Storage Research
- 📖 **HVAC Standards:** ASHRAE 62.1 & 90.1 Guidelines
- 📖 **Machine Learning:** scikit-learn Documentation & Scikit Learn Tutorials
- 📖 **Time Series:** Feature Engineering for Temporal Data Analysis
- 📖 **Energy Economics:** Electricity Pricing in Deregulated Markets

---

## 📜 License

MIT License - Free for commercial and personal use

---

## 🙏 Acknowledgments

- **PCM Research:** NREL (National Renewable Energy Laboratory)
- **HVAC Dynamics:** ASHRAE (American Society of Heating, Refrigerating and Air-Conditioning Engineers)
- **Energy Data:** Indian Electricity Regulatory Commission (CERC)
- **ML Framework:** scikit-learn Open Source Community

---

## 🤝 Contributing

Found a bug or have an idea? Contributions are welcome! 

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact & Support

- 💬 **Questions?** Check the inline code comments - everything is documented
- 🐛 **Bug Report?** Create an issue with reproduction steps
- 💡 **Feature Request?** Open a discussion thread

---

<p align="center">
  <strong>Ready to revolutionize HVAC efficiency?</strong>

  ```bash
  streamlit run dashboard/app.py
  ```
  
  <br><center>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square" alt="Status"/>
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
</p>
