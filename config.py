"""
Configuration settings for Driver Behavior Risk Prediction System
"""

# ==========================
# Model Paths
# ==========================

MODEL_PATH = "models/driver_risk_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# ==========================
# Dataset Path
# ==========================

DATASET_PATH = "data/dataset_7M.csv"

# ==========================
# Feature Names
# ==========================

FEATURE_COLUMNS = [
    "AccX",
    "AccY",
    "AccZ",
    "GyroX",
    "GyroY",
    "GyroZ",
    "motion_magnitude",
    "rotation_magnitude",
    "driving_intensity",
    "harsh_braking",
    "sharp_turning"
]

# ==========================
# Model Information
# ==========================

MODEL_NAME = "Random Forest Classifier"
MODEL_ACCURACY = "99.90%"