import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time

from utils.feature_engineering import create_features
from utils.prediction import predict_driver_behavior
from utils.logger import logger

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="Driver Risk Prediction",
    page_icon="🚗",
    layout="wide"
)

# =====================================
# Dark Mode Toggle
# =====================================

dark_mode = st.sidebar.toggle("🌙 Enable Dark Mode")

if dark_mode:
    st.markdown("""
        <style>

        .stApp {
            background-color: #0E1117;
            color: white;
        }

        header {
            visibility: hidden;
        }

        .block-container {
            padding-top: 1rem;
        }

        section[data-testid="stSidebar"] {
            background-color: #111827;
        }

        label, p, span, div {
            color: white !important;
        }

        div.stButton > button {
            background-color: #2563EB;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px 20px;
        }

        div.stButton > button:hover {
            background-color: #1D4ED8;
            transform: scale(1.05);
            transition: 0.2s;
        }

        </style>
    """, unsafe_allow_html=True)

# =====================================
# Load model (for feature importance)
# =====================================

@st.cache_resource
def load_model():
    return joblib.load("models/driver_risk_model.pkl")

rf_model = load_model()

# =====================================
# Sidebar
# =====================================

st.sidebar.title("🚗 Driver Risk Analytics")
st.sidebar.markdown("### Model Details")
st.sidebar.write("Model: Random Forest Classifier")
st.sidebar.metric("Test Accuracy", "99.90%")
st.sidebar.markdown("---")

st.sidebar.write("Use Cases:")
st.sidebar.write("• Insurance Risk Profiling")
st.sidebar.write("• Fleet Monitoring")
st.sidebar.write("• Driver Safety Scoring")

# =====================================
# Header
# =====================================

st.title("🚗 Driver Behavior Risk Prediction System")
st.write("AI-powered real-time driver risk analysis dashboard")

st.markdown("---")

# =====================================
# Layout
# =====================================

input_col, output_col = st.columns([1,1])

# =====================================
# Sensor Inputs
# =====================================

with input_col:

    st.subheader("Sensor Inputs")

    accx = st.number_input("Acceleration X", value=0.0)
    accy = st.number_input("Acceleration Y", value=0.0)
    accz = st.number_input("Acceleration Z", value=0.0)

    gyrox = st.number_input("Gyro X", value=0.0)
    gyroy = st.number_input("Gyro Y", value=0.0)
    gyroz = st.number_input("Gyro Z", value=0.0)

    analyze = st.button("🚀 Analyze Driving Behavior")

# =====================================
# Risk Assessment
# =====================================

with output_col:

    st.subheader("Risk Assessment")

    if analyze:

        logger.info("Driver prediction requested")

        with st.spinner("Analyzing sensor signals..."):

            time.sleep(1)

            features = create_features(
                accx, accy, accz,
                gyrox, gyroy, gyroz
            )

            label, confidence = predict_driver_behavior(features)

            # Risk mapping
            if label == "AGGRESSIVE":
                score = 90
                level = "HIGH RISK"
                color = "red"
                st.error(f"⚠️ {label} — {level}")

            elif label == "NORMAL":
                score = 50
                level = "MEDIUM RISK"
                color = "orange"
                st.warning(f"⚠️ {label} — {level}")

            else:
                score = 10
                level = "LOW RISK"
                color = "green"
                st.success(f"✅ {label} — {level}")

            st.write(f"Confidence Level: {confidence:.2f}%")

            # Animated Risk Gauge
            gauge_placeholder = st.empty()

            for i in range(0, score + 1, 5):

                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=i,
                    title={'text': "Risk Score"},
                    gauge={
                        'axis': {'range': [0,100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range':[0,30],'color':'green'},
                            {'range':[30,70],'color':'orange'},
                            {'range':[70,100],'color':'red'}
                        ]
                    }
                ))

                gauge_placeholder.plotly_chart(gauge, use_container_width=True)
                time.sleep(0.03)

            # Driver Safety Score
            st.markdown("### 🛡 Driver Safety Score")

            safety_score = 100 - score

            st.progress(int(safety_score))

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Safety Score", f"{safety_score}/100")

            with col2:
                st.metric("Risk Score", f"{score}/100")

            if safety_score >= 80:
                st.success("Excellent Driving Behavior 🚗")

            elif safety_score >= 50:
                st.warning("Moderate Driving Behavior ⚠")

            else:
                st.error("Unsafe Driving Detected 🚨")
# =====================================
# Feature Importance
# =====================================

if analyze:

    st.markdown("---")
    st.subheader("📈 Feature Importance Analysis")

    importances = rf_model.feature_importances_

    features_names = [
        "AccX","AccY","AccZ",
        "GyroX","GyroY","GyroZ",
        "motion_magnitude",
        "rotation_magnitude",
        "driving_intensity",
        "harsh_braking",
        "sharp_turning"
    ]

    importance_df = pd.DataFrame({
        "Feature": features_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="viridis"
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("© 2025 Driver Risk Prediction System")