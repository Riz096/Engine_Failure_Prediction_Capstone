%%writefile engine_data/deployment/app.py

import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Engine Failure Prediction",
    layout="centered"
)


# ==============================
# LOAD MODEL (WITH CACHING)
# ==============================

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="Rizwan9/Engine_Failure_Model",
            filename="best_engine_model.pkl"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_model()


# ==============================
# UI
# ==============================

st.title("🔧 Engine Failure Prediction System")

st.write("""
This application predicts whether an engine is likely to **fail** based on sensor readings.

Helps maintenance teams take preventive action.
""")


# ==============================
# INPUTS
# ==============================

engine_rpm = st.number_input("Engine RPM", 500, 5000, 1500)
lub_oil_pressure = st.number_input("Lubrication Oil Pressure", 0.0, 10.0, 3.5)
fuel_pressure = st.number_input("Fuel Pressure", 0.0, 10.0, 4.0)
coolant_pressure = st.number_input("Coolant Pressure", 0.0, 10.0, 2.5)
lub_oil_temp = st.number_input("Lubrication Oil Temperature", 50.0, 150.0, 90.0)
coolant_temp = st.number_input("Coolant Temperature", 50.0, 150.0, 85.0)


# ==============================
# PREDICTION
# ==============================

if st.button("Predict Engine Condition"):

    if model is None:
        st.error("Model not loaded. Please check deployment.")
    else:
        try:
            input_data = pd.DataFrame([{
                "Engine rpm": engine_rpm,
                "Lub oil pressure": lub_oil_pressure,
                "Fuel pressure": fuel_pressure,
                "Coolant pressure": coolant_pressure,
                "Lub oil temp": lub_oil_temp,
                "Coolant temp": coolant_temp
            }])

            prediction = model.predict(input_data)[0]

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("Engine Failure Likely – Maintenance Required")
            else:
                st.success("Engine Operating Normally")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
