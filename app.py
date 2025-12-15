import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# Page setup
st.set_page_config(page_title="India Air Quality App", layout="wide")

st.title(" India Air Quality Prediction App")
st.markdown(
    "This interactive app predicts **PM2.5 concentration** based on pollutant levels using a trained Machine Learning model."
)

# Load model
model = joblib.load("models/best_model.pkl")
mae = 10.37  # Model MAE

# ===================== SIDEBAR =====================
st.sidebar.header(" Input Controls")

city = st.sidebar.selectbox(
    "Select City Scenario",
    ["Custom", "Delhi", "Mumbai", "Bengaluru"]
)

# Preset values
presets = {
    "Delhi": [250, 80, 40, 2.5, 4, 10],
    "Mumbai": [150, 50, 25, 1.5, 2, 6],
    "Bengaluru": [90, 35, 15, 1.0, 1, 4]
}

if city != "Custom":
    pm10, no2, so2, co, benzene, toluene = presets[city]
else:
    pm10 = st.sidebar.slider("PM10 (µg/m³)", 0.0, 1000.0, 100.0)
    no2 = st.sidebar.slider("NO2 (µg/m³)", 0.0, 300.0, 30.0)
    so2 = st.sidebar.slider("SO2 (µg/m³)", 0.0, 200.0, 20.0)
    co = st.sidebar.slider("CO (mg/m³)", 0.0, 50.0, 1.0)
    benzene = st.sidebar.slider("Benzene (µg/m³)", 0.0, 50.0, 1.0)
    toluene = st.sidebar.slider("Toluene (µg/m³)", 0.0, 100.0, 5.0)

# ===================== PREDICTION =====================
st.header(" Prediction")

if st.button("Predict PM2.5"):
    input_data = np.array([[pm10, no2, so2, co, benzene, toluene]])
    prediction = model.predict(input_data)[0]

    # Category logic
    if prediction < 50:
        category, color = "Good", "green"
        advisory = "Air quality is satisfactory."
    elif prediction < 100:
        category, color = "Moderate", "gold"
        advisory = "Acceptable air quality for most individuals."
    elif prediction < 250:
        category, color = "Poor", "orange"
        advisory = "Unhealthy for sensitive groups."
    else:
        category, color = "Hazardous", "red"
        advisory = "Avoid outdoor activities."

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"### PM2.5: {prediction:.2f} µg/m³")
        st.write(
            f"**Confidence Range:** {prediction - mae:.2f} – {prediction + mae:.2f} µg/m³"
        )
        st.markdown(f"### Category: **:{color}[{category}]**")
        st.info(advisory)

    # ===================== GAUGE =====================
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={"text": "PM2.5 Level"},
            gauge={
                "axis": {"range": [0, 400]},
                "steps": [
                    {"range": [0, 50], "color": "green"},
                    {"range": [50, 100], "color": "yellow"},
                    {"range": [100, 250], "color": "orange"},
                    {"range": [250, 400], "color": "red"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "value": prediction
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# ===================== INFO =====================
with st.expander(" Interpretation Guide"):
    st.markdown("""
    - **PM2.5 < 50** → Good
    - **50 – 100** → Moderate
    - **100 – 250** → Poor
    - **> 250** → Hazardous

    PM2.5 particles can penetrate deep into the lungs and bloodstream, causing serious health issues.
    """)
