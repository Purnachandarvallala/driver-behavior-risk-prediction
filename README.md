Driver Behavior Risk Prediction System



AI-powered real-time driver risk analysis dashboard built using Machine Learning and Streamlit.



This system analyzes vehicle sensor signals (accelerometer and gyroscope) to detect risky driving behavior and classify drivers into risk categories.



The dashboard provides interactive visualizations including risk meters, safety score cards, and feature importance charts.



Dashboard Preview



Risk Prediction Example



Driver Safety Score



Feature Importance Analysis



Project Overview



Driver behavior monitoring is important for several real-world applications such as:



Insurance Risk Profiling

Fleet Monitoring

Driver Safety Scoring

Vehicle Telematics Systems



This project builds a machine learning pipeline that processes vehicle sensor data and predicts driving risk levels using a Random Forest classifier. The results are visualized in an interactive Streamlit dashboard.



Machine Learning Pipeline



Sensor Data

↓

Feature Engineering

↓

Data Scaling

↓

Random Forest Model

↓

Risk Prediction

↓

Interactive Dashboard Visualization



Key Features



AI-based driving behavior prediction



Interactive Streamlit dashboard



Animated risk meter gauge



Driver safety score card



Feature importance visualization



Dark mode dashboard UI



Modular production code structure



Prediction logging system



Dataset



The dataset contains vehicle sensor readings from accelerometer and gyroscope sensors.



Features used in the model include:



Acceleration X

Acceleration Y

Acceleration Z

Gyroscope X

Gyroscope Y

Gyroscope Z



Additional engineered features:



Motion Magnitude

Rotation Magnitude

Driving Intensity

Harsh Braking Detection

Sharp Turning Detection



Model Performance



Model Used

Random Forest Classifier



Test Accuracy

99.90%



Project Structure



Driver-Behavior-Risk-Prediction-System



app.py

config.py

requirements.txt

README.md

.gitignore



models/

driver\_risk\_model.pkl

scaler.pkl

label\_encoder.pkl



data/

dataset\_7M.csv



notebooks/

bmw\_project.ipynb



utils/

feature\_engineering.py

prediction.py

logger.py



logs/



images/



Installation



Clone the repository



git clone https://github.com/YOUR\_USERNAME/driver-behavior-risk-prediction.git



Navigate to the project directory



cd driver-behavior-risk-prediction



Install dependencies



pip install -r requirements.txt



Run the Streamlit dashboard



streamlit run app.py



The dashboard will open in your browser at



http://localhost:8501



Example Prediction



Input Sensor Values



AccX = -2

AccY = 1.5

AccZ = 2

GyroX = 3

GyroY = 2

GyroZ = 4



Predicted Output



Driving Behavior: Aggressive



Risk Level: High Risk



Driver Safety Score: 10/100



Technologies Used



Python

Scikit-Learn

Streamlit

Plotly

Pandas

NumPy

Joblib



Future Improvements



Real-time vehicle sensor streaming



Driver history analytics dashboard



Explainable AI (SHAP analysis)



Fleet monitoring system



Downloadable driver risk reports



Author



Purnachandar Vallala



Machine Learning Project



