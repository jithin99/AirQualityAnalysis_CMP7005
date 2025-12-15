import os
import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# --------------------------------------------------
# CITY COORDINATES (NO LAT/LON IN DATASET)
# --------------------------------------------------
CITY_COORDS = {
    "Delhi": [28.6139, 77.2090],
    "Mumbai": [19.0760, 72.8777],
    "Kolkata": [22.5726, 88.3639],
    "Chennai": [13.0827, 80.2707],
    "Bengaluru": [12.9716, 77.5946],
    "Hyderabad": [17.3850, 78.4867],
    "Pune": [18.5204, 73.8567],
    "Ahmedabad": [23.0225, 72.5714],
    "Jaipur": [26.9124, 75.7873],
    "Lucknow": [26.8467, 80.9462]
}

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Air Quality Analysis – CMP7005",
    layout="wide"
)

st.title("🌫️ Air Quality Analysis & Prediction")
st.markdown("CMP7005 PRACTICAL – Streamlit Cloud Deployment")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Data/merged_data.csv")

df = load_data()

# --------------------------------------------------
# LOAD MODEL (GOOGLE DRIVE)
# --------------------------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1N317Atsm71Is04H_P711V3Dk-jr5y1ou"
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading ML model (one-time)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["🔮 PM2.5 Prediction", "📊 Dataset Overview", "🗺️ Air Quality Map"]
)

# ==================================================
# 🔮 TAB 1 – PREDICTION
# ==================================================
with tab1:
    st.subheader("Predict PM2.5 Concentration")

    col1, col2 = st.columns(2)

    with col1:
        so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=10.0)
        no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=20.0)
        co = st.number_input("CO (mg/m³)", min_value=0.0, value=1.0)

    with col2:
        o3 = st.number_input("O₃ (µg/m³)", min_value=0.0, value=30.0)
        pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=50.0)
        nh3 = st.number_input("NH₃ (µg/m³)", min_value=0.0, value=15.0)

    if st.button("🔮 Predict PM2.5"):
        X = np.array([[so2, no2, co, o3, pm10, nh3]])
        pred = model.predict(X)[0]

        st.success(f"🌟 Predicted PM2.5: **{pred:.2f} µg/m³**")

        if pred <= 60:
            st.info("🟢 Air Quality: Good")
        elif pred <= 120:
            st.warning("🟡 Air Quality: Moderate")
        else:
            st.error("🔴 Air Quality: Poor")

# ==================================================
# 📊 TAB 2 – DATASET OVERVIEW
# ==================================================
with tab2:
    st.subheader("Dataset Overview")

    st.write("### Sample Records")
    st.dataframe(df.head())

    st.write("### Dataset Statistics")
    st.dataframe(df.describe())

    st.write("### Column Names")
    st.code(", ".join(df.columns))

# ==================================================
# 🗺️ TAB 3 – AIR QUALITY MAP (FIXED)
# ==================================================
with tab3:
    st.subheader("India Air Quality Map (PM2.5)")

    # Average PM2.5 per city
    city_pm = df.groupby("City")["PM2.5"].mean().reset_index()

    # Base map
    m = folium.Map(location=[22.5, 80.0], zoom_start=5)

    for _, row in city_pm.iterrows():
        city = row["City"]
        pm25 = row["PM2.5"]

        if city in CITY_COORDS:
            lat, lon = CITY_COORDS[city]

            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                popup=f"{city}<br>PM2.5: {pm25:.2f}",
                color="red" if pm25 > 60 else "orange" if pm25 > 30 else "green",
                fill=True,
                fill_opacity=0.7
            ).add_to(m)

    st_folium(m, width=1000, height=500)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("👨‍🎓 **Student:** Jithin")
st.markdown("📘 **Course:** CMP7005 – Air Quality Analysis")
st.markdown("☁️ Deployed on Streamlit Cloud")
