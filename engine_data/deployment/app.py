import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ==============================
# Load Model from Hugging Face
# ==============================

model_path = hf_hub_download(
    repo_id="Rizwan9/Engine_Failure_Model",
    filename="best_engine_model.pkl"
)

model = joblib.load(model_path)

# ==============================
# Streamlit UI
# ==============================

st.title("Engine Failure Prediction System")

st.write("""
This application predicts whether an engine is likely to **fail** based on sensor readings.
It can help maintenance teams detect potential engine problems early.
""")

# ==============================
# User Inputs (Engine Sensors)
# ==============================

engine_rpm = st.number_input(
    "Engine RPM",
    min_value=500,
    max_value=5000,
    value=1500
)

lub_oil_pressure = st.number_input(
    "Lubrication Oil Pressure",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

fuel_pressure = st.number_input(
    "Fuel Pressure",
    min_value=0.0,
    max_value=10.0,
    value=4.0
)

coolant_pressure = st.number_input(
    "Coolant Pressure",
    min_value=0.0,
    max_value=10.0,
    value=2.5
)

lub_oil_temp = st.number_input(
    "Lubrication Oil Temperature",
    min_value=50.0,
    max_value=150.0,
    value=90.0
)

coolant_temp = st.number_input(
    "Coolant Temperature",
    min_value=50.0,
    max_value=150.0,
    value=85.0
)

# ==============================
# Create Input DataFrame
# ==============================

input_data = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "Lub oil temp": lub_oil_temp,
    "Coolant temp": coolant_temp
}])

# ==============================
# Prediction
# ==============================

if st.button("Predict Engine Condition"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Engine Failure Likely – Maintenance Required")
    else:
        st.success("Engine Operating Normally")
