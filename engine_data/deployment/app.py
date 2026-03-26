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
# LOAD MODEL
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


with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.stop()


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

    try:
        # FIXED COLUMN ORDER
        columns = [
            "Engine rpm",
            "Lub oil pressure",
            "Fuel pressure",
            "Coolant pressure",
            "Lub oil temp",
            "Coolant temp"
        ]

        input_data = pd.DataFrame([[
            engine_rpm,
            lub_oil_pressure,
            fuel_pressure,
            coolant_pressure,
            lub_oil_temp,
            coolant_temp
        ]], columns=columns)

        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        # Optional probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0][1]
            st.write(f"Failure Probability: {prob:.2f}")

        if prediction == 1:
            st.error("Engine Failure Likely – Maintenance Required")
        else:
            st.success("Engine Operating Normally")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
