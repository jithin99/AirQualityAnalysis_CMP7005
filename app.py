
import streamlit as st
import joblib
import numpy as np

# Page setup
st.set_page_config(page_title="Air Quality App", layout="wide")
st.title(" India Air Quality Analysis")

st.markdown("""
I built a simple Streamlit app where users enter pollutant values and the model predicts the PM2.5 level. This makes the model more interactive instead of just running in a notebook**.
""")

# Load model
joblib.load("Models/best_model.pkl")


st.header("Enter Pollutant Levels")

# Inputs
pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=1000.0, value=100.0)
no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, max_value=300.0, value=30.0)
so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, max_value=200.0, value=20.0)
co = st.number_input("CO (mg/m³)", min_value=0.0, max_value=50.0, value=1.0)
benzene = st.number_input("Benzene (µg/m³)", min_value=0.0, max_value=50.0, value=1.0)
toluene = st.number_input("Toluene (µg/m³)", min_value=0.0, max_value=100.0, value=5.0)

input_data = np.array([[pm10, no2, so2, co, benzene, toluene]])
prediction = model.predict(input_data)[0]
mae = 10.37  # From tuned model evaluation

# Output
st.header("Prediction Result")
st.success(f"Estimated PM2.5: **{prediction:.2f} µg/m³**")
st.write(f"Confidence range: **{prediction - mae:.2f} to {prediction + mae:.2f} µg/m³** (± MAE)")

# Air quality category
if prediction < 50:
    category = "Good"
    color = "🟢"
    advisory = "Air quality is satisfactory and poses little or no risk."
elif prediction < 100:
    category = "Moderate"
    color = "🟡"
    advisory = "Acceptable air quality, but may affect sensitive individuals."
elif prediction < 250:
    category = "Poor"
    color = "🟠"
    advisory = "Air quality is unhealthy for sensitive groups. Reduce prolonged outdoor exposure."
else:
    category = "Hazardous"
    color = "🔴"
    advisory = "Health warnings of emergency conditions. Avoid outdoor activities."

st.markdown(f"### Air Quality Category: {color} **{category}**")
st.info(advisory)

# Interpretation
st.header(" Interpretation Guide")
st.markdown("""
- **PM2.5 < 50** → Good
- **50 ≤ PM2.5 < 100** → Moderate
- **100 ≤ PM2.5 < 250** → Poor
- **PM2.5 ≥ 250** → Hazardous

PM2.5 refers to fine particulate matter that can penetrate deep into the lungs and bloodstream.
This model helps simulate pollution scenarios and assess air quality risk based on key pollutants.
""")
