# Driver Behavior Risk Prediction
### Real-Time Driving Risk Classification · Random Forest · Streamlit · 7M Rows

**Live Dashboard →** [https://driver-behavior-risk-prediction-xp7dugajuvgyokuzfpx5hw.streamlit.app/]  
**Author →** Purnachandar Vallala | MSc Data Science · Germany  
**LinkedIn →** [www.linkedin.com/in/vallala-purnachandar-051314226]

---

## Background

Usage-based insurance (UBI) is growing fast across Europe. Insurers 
including HUK-COBURG and Allianz are increasingly pricing premiums 
from actual driving behaviour rather than demographic proxies. The 
technical foundation of every UBI system is the same: take raw inertial 
sensor data from a vehicle and classify how that driver actually drives.

This project builds that pipeline end-to-end. Raw accelerometer and 
gyroscope readings go in. A risk classification and interpretable 
safety score come out. The model was trained on 7 million sensor 
readings and achieves 99.90% test accuracy across three driving 
behaviour classes.

The broader applications go beyond insurance — fleet management, 
driver coaching systems, road safety research, and automotive 
telematics all rely on the same core classification task.

---

## The Problem

A vehicle's accelerometer and gyroscope generate six continuous 
sensor channels while driving:
```
AccX · AccY · AccZ → linear acceleration across three axes
GyroX · GyroY · GyroZ → rotational velocity across three axes
```

These six numbers, sampled continuously, encode everything about 
how a driver behaves — smooth acceleration, harsh braking, sharp 
cornering, erratic lane changes. The classification challenge is 
extracting meaningful driving patterns from this raw signal and 
assigning them accurately to risk categories in real time.

---

## Dataset

| Metric | Value |
|--------|-------|
| Total rows | 7,000,000 |
| Raw sensor features | 6 |
| Engineered features | 5 |
| Total model features | 11 |
| Target classes | Aggressive · Normal · Slow |
| Model file | driver_risk_model.pkl (joblib) |

Seven million rows makes this a genuinely large-scale classification 
task. Training on this volume required deliberate decisions around 
memory management and processing efficiency — not considerations 
that arise on standard toy datasets.

---

## Feature Engineering

Six raw channels are not sufficient on their own. Driving behaviour 
is a composite phenomenon — harsh braking involves multiple axes 
simultaneously, and sharp turning is only identifiable when 
gyroscope and accelerometer signals are combined. Five derived 
features were engineered to capture these interactions:

| Feature | Computation | Physical Meaning |
|---------|-------------|-----------------|
| Motion Magnitude | √(AccX² + AccY² + AccZ²) | Total acceleration intensity across all axes |
| Rotation Magnitude | √(GyroX² + GyroY² + GyroZ²) | Total rotational intensity across all axes |
| Driving Intensity | Motion Magnitude × Rotation Magnitude | Combined aggressiveness signal — high when both acceleration and rotation are elevated simultaneously |
| Harsh Braking | AccX below defined threshold | Captures sudden forward deceleration events characteristic of emergency or aggressive stops |
| Sharp Turning | GyroZ above defined threshold | Captures high yaw-rate events characteristic of aggressive cornering |

These features are computed in `utils/feature_engineering.py` and 
applied identically during training and at inference time. This is 
intentional — keeping feature engineering in a shared module 
eliminates training-serving skew, a common failure mode in 
production ML systems where models perform well in notebooks 
but degrade in deployment.

---

## Model

**Random Forest Classifier | Test Accuracy: 99.90%**

Three properties of Random Forest made it the right choice for 
this specific problem:

**Non-linear sensor interactions.** The relationship between 
raw sensor channels and driving behaviour is not linear. 
A high AccX reading means something different depending on 
simultaneous GyroZ values. Tree-based models capture these 
conditional interactions without requiring explicit interaction 
terms in the feature set.

**Robustness to sensor noise.** Real inertial sensor data 
contains measurement noise — vibration from road surface, 
sensor drift, signal interference. Ensemble averaging across 
the forest smooths out noise-driven misclassifications that 
would consistently affect a single decision tree.

**Native feature importance.** The model produces interpretable 
feature importance scores directly from the ensemble structure. 
This feeds the dashboard's feature importance panel without 
requiring a post-hoc explanation method. What the model learned 
is visible in the interface — not treated as a black box.

### Risk Classification Output

| Predicted Class | Risk Level | Safety Score | Visual Indicator |
|----------------|-----------|--------------|-----------------|
| AGGRESSIVE | High Risk | 10 / 100 | Red zone (70–100) |
| NORMAL | Medium Risk | 50 / 100 | Orange zone (30–70) |
| SLOW | Low Risk | 90 / 100 | Green zone (0–30) |

---

## ML Pipeline
```
Raw Sensor Data
(AccX, AccY, AccZ, GyroX, GyroY, GyroZ)
        ↓
Feature Engineering
(5 derived features → 11 total features)
        ↓
StandardScaler
(zero mean, unit variance — consistent with training)
        ↓
Random Forest Classifier
(trained on 7M rows, 99.90% test accuracy)
        ↓
LabelEncoder → Risk Class + Confidence Score
        ↓
Streamlit Dashboard
(animated gauge, safety score, feature importance)
        ↓
Prediction Logger
(audit trail for every classification)
```

---

## Dashboard

The dashboard is structured around a single prediction workflow: 
enter sensor values, trigger analysis, receive a risk classification 
with visual feedback, then inspect which features drove the result.

**Sensor input panel** — six numerical inputs (AccX, AccY, AccZ, 
GyroX, GyroY, GyroZ) that simulate a real-time sensor stream. 
The input structure mirrors what an OBD-II or CAN bus interface 
would deliver in a production deployment.

**Animated risk gauge** — Plotly Indicator that fills 
progressively from 0 to the predicted risk score, with three 
colour zones matching the classification thresholds: green 
(0–30), orange (30–70), red (70–100). The animation is rendered 
frame-by-frame to simulate real-time signal processing.

**Safety score card** — dual metric display showing safety score 
and risk score side by side with a Streamlit progress bar. 
Score is derived as 100 minus the risk score, giving an 
intuitive driver performance metric for non-technical users.

**Feature importance chart** — horizontal bar chart showing 
the Random Forest's feature importances for all 11 input 
features, ranked by contribution. Rendered using Plotly Express 
with viridis colour scale. This panel answers "why was this 
driver classified as aggressive?" in an interpretable way.

**Dark mode toggle** — sidebar switch that applies a dark 
CSS theme to the entire application, including sidebar, 
metrics, and button states.

**Confidence display** — model confidence percentage shown 
alongside the risk classification, giving users a signal 
about prediction certainty.

---

## Example Prediction

**Input sensor values:**

| Sensor | Value |
|--------|-------|
| AccX | -2.0 |
| AccY | 1.5 |
| AccZ | 2.0 |
| GyroX | 3.0 |
| GyroY | 2.0 |
| GyroZ | 4.0 |

**Derived features computed at inference:**

| Feature | Computed Value |
|---------|---------------|
| Motion Magnitude | √(4 + 2.25 + 4) = 3.18 |
| Rotation Magnitude | √(9 + 4 + 16) = 5.39 |
| Driving Intensity | 3.18 × 5.39 = 17.14 |
| Harsh Braking | True (AccX = -2.0 below threshold) |
| Sharp Turning | True (GyroZ = 4.0 above threshold) |

**Output:**

| Field | Value |
|-------|-------|
| Driving Behavior | AGGRESSIVE |
| Risk Level | HIGH RISK |
| Safety Score | 10 / 100 |
| Risk Score | 90 / 100 |

---

## Business Recommendations

The model output enables five concrete downstream applications 
across the insurance, fleet, and automotive sectors.

**1. Usage-Based Insurance Premium Calculation**

The three-class output (Aggressive / Normal / Slow) maps 
directly to premium adjustment factors. An aggressive 
driver identified by the system presents measurably higher 
claim risk. Insurers can apply per-trip or per-month 
risk multipliers: Aggressive +25–40% premium loading, 
Normal baseline, Slow −5–10% discount. German insurers 
moving toward UBI models — HUK-COBURG's Telematik Plus 
product is an example — need exactly this classification 
layer as their pricing engine.

**2. Fleet Driver Coaching and Performance Reporting**

For logistics operators and commercial fleet managers, 
aggregating predictions across all trips per driver 
produces a weekly or monthly driver risk profile. 
A driver whose trips score 70%+ Aggressive classifications 
needs immediate coaching intervention — specific to the 
behaviours the feature importance panel identifies 
(e.g., consistently high GyroZ indicating cornering 
aggression). This is more actionable than a generic 
"your score decreased" notification.

**3. Real-Time In-Vehicle Alerts**

Connecting the inference pipeline to a live OBD-II 
sensor stream enables real-time in-vehicle feedback. 
When driving intensity exceeds the Aggressive threshold, 
the system triggers an alert — visual, audio, or 
haptic — within the vehicle. This application is 
relevant for delivery drivers, taxi operators, 
and driver training systems. The current architecture 
supports this with a streaming input wrapper around 
the existing `predict_driver_behavior` function.

**4. Road Safety Research and Urban Planning**

At population scale, aggregated anonymised 
classifications across many drivers on the same 
road segments identify genuinely dangerous road 
sections — not from accident reports after the 
fact, but from the distribution of Aggressive 
classifications before accidents occur. City 
planners and road safety authorities (Verkehrsbehörden) 
can use this signal to prioritise infrastructure 
interventions: speed camera placement, road 
geometry improvements, signage changes.

**5. Autonomous Vehicle Training Data Validation**

The Aggressive, Normal, and Slow classifications 
provide structured labels for sensor sequences 
that autonomous vehicle developers need. 
Labelled examples of harsh braking events 
(high AccX, Harsh Braking = True) and sharp 
turning events (high GyroZ, Sharp Turning = True) 
are exactly the edge cases AV systems must 
handle reliably. Companies including Continental, 
Bosch, and ZF — all with major German operations — 
maintain ongoing demand for validated sensor 
datasets with behaviour labels.

---

## Code Structure

The repository is organised as a modular production-style 
codebase. The core logic is separated into independent 
modules rather than kept in a single notebook or script, 
which mirrors how ML systems are structured in 
professional engineering environments:
```
driver-behavior-risk-prediction/
│
├── app.py                       # Streamlit dashboard (256 lines)
├── config.py                    # Centralised paths and constants
├── requirements.txt             # Python dependencies
│
├── models/
│   ├── driver_risk_model.pkl    # Trained Random Forest (joblib)
│   ├── scaler.pkl               # StandardScaler fitted on training data
│   └── label_encoder.pkl        # Class label encoder
│
├── notebooks/
│   └── bmw_project.ipynb        # Full training pipeline and EDA
│
├── utils/
│   ├── feature_engineering.py   # Shared feature creation module
│   ├── prediction.py            # Inference wrapper with confidence scoring
│   └── logger.py                # Prediction audit logging
│
├── images/                      # Dashboard screenshots
└── .gitignore
```

The `config.py` file centralises all file paths, feature 
column names, and model metadata. This means changing 
a model path or adding a feature only requires editing 
one file — not hunting through app.py and training 
notebooks separately.

The `utils/` separation is the most important structural 
decision. `feature_engineering.py` is imported by both 
the training notebook and `app.py`. Any change to a 
feature definition automatically applies to both 
training and inference — there is no risk of the 
two drifting apart silently.

---

## Real-World Applications

| Sector | Application | Relevant German Companies |
|--------|-------------|--------------------------|
| Insurance | UBI premium pricing engine | HUK-COBURG, Allianz, ERGO |
| Logistics | Fleet driver performance monitoring | DHL, DB Schenker, Kuehne+Nagel |
| Automotive | ADAS and AV training data labelling | Continental, Bosch, ZF, BMW |
| Public sector | Road safety signal for urban planning | Verkehrsbehörden, ADAC |
| Mobility | Ride-hailing driver safety scoring | FREE NOW, Sixt |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| scikit-learn | Random Forest, StandardScaler, LabelEncoder |
| joblib | Model and scaler serialisation |
| pandas | Data loading and manipulation |
| numpy | Numerical feature computation |
| plotly | Animated gauge and feature importance charts |
| streamlit | Dashboard deployment and UI |

---

## Run Locally
```bash
git clone https://github.com/Purnachandarvallala/driver-behavior-risk-prediction.git
cd driver-behavior-risk-prediction
pip install -r requirements.txt
streamlit run app.py
```

Dashboard opens at `http://localhost:8501`

---

## What Could Be Extended

Three extensions would meaningfully deepen the technical scope:

**Per-prediction SHAP values** — the current feature importance 
panel shows global importances from the Random Forest ensemble. 
SHAP would show exactly which features pushed a specific prediction 
toward Aggressive rather than Normal, making the explanation local 
and per-instance rather than averaged across the training set.

**Real-time sensor streaming via Kafka** — replacing manual 
number input with a Kafka consumer connected to a simulated 
OBD-II stream would make this a genuine real-time monitoring 
system. The `predict_driver_behavior` function in `utils/prediction.py` 
is already structured as a standalone callable — adding a 
streaming input is an architectural extension, not a rewrite.

**Driver history and trip aggregation** — storing predictions 
to a database and building a second dashboard page that 
aggregates per-driver risk scores over time would make this 
a complete fleet monitoring tool rather than a single-trip 
predictor.

---

## Author

**Purnachandar Vallala**  
MSc Data Science · Germany  
[GitHub](https://github.com/Purnachandarvallala) ·
[LinkedIn](www.linkedin.com/in/vallala-purnachandar-051314226)

*Available for Werkstudent and full-time Data Science
and Machine Learning positions in Germany.*
