import joblib
import pandas as pd

# Load trained model artifacts
model = joblib.load("models/driver_risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")

columns = [
    "AccX","AccY","AccZ",
    "GyroX","GyroY","GyroZ",
    "motion_magnitude",
    "rotation_magnitude",
    "driving_intensity",
    "harsh_braking",
    "sharp_turning"
]

def predict_driver_behavior(features):

    df = pd.DataFrame([features], columns=columns)

    scaled = scaler.transform(df)

    prediction = model.predict(scaled)

    label = encoder.inverse_transform(prediction)[0]

    probabilities = model.predict_proba(scaled)[0]

    confidence = max(probabilities) * 100

    return label, confidence