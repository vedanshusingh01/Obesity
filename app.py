import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("final_model.pkl")

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Obesity Predictor", layout="centered")

st.title("🧠 Biomarker-Based Obesity Classification")
st.write("Enter clinical biomarker values to predict obesity category")

# ==============================
# INPUT SECTION
# ==============================

age = st.slider("Age", 10, 80, 30)

gender = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

glucose = st.number_input("Glucose (mg/dL)", 70, 200, 100)
insulin = st.number_input("Insulin (µU/mL)", 2, 50, 10)
uric_acid = st.number_input("Uric Acid (mg/dL)", 1.0, 15.0, 5.5, 0.1)

trig = st.number_input("Triglycerides", 50, 400, 150)
hdl = st.number_input("HDL Cholesterol", 20, 100, 50)

# ==============================
# FEATURE ENGINEERING
# ==============================

homa_ir = (insulin * glucose) / 405
tg_hdl = trig / hdl
insulin_tg = insulin * trig
glucose_tg = glucose * trig

# ==============================
# CREATE INPUT DATAFRAME
# ==============================

input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Glucose": [glucose],
    "Insulin": [insulin],
    "HDL": [hdl],
    "Triglycerides": [trig],
    "Uric_Acid": [uric_acid],
    "HOMA_IR": [homa_ir],
    "TG_HDL_Ratio": [tg_hdl],
    "Insulin_TG": [insulin_tg],
    "Glucose_TG": [glucose_tg]
})

# ==============================
# ENSURE COLUMN ORDER (CRITICAL FIX)
# ==============================

expected_columns = getattr(
    model,
    "feature_names_",
    [
        "Age",
        "Gender",
        "Glucose",
        "Insulin",
        "HDL",
        "Triglycerides",
        "Uric_Acid",
        "HOMA_IR",
        "TG_HDL_Ratio",
        "Insulin_TG",
        "Glucose_TG",
    ],
)

input_data = input_data.reindex(columns=expected_columns)

# ==============================
# PREDICTION
# ==============================

if st.button("🔍 Predict"):

    raw_prediction = model.predict(input_data)
    prediction = int(np.asarray(raw_prediction).reshape(-1)[0])

    raw_probabilities = model.predict_proba(input_data)
    probabilities = np.asarray(raw_probabilities).reshape(-1)

    labels = {
        1: "Normal",
        2: "Overweight",
        3: "Obese"
    }

    st.success(f"Prediction: {labels[prediction]}")

    # ==============================
    # PROBABILITY DISPLAY
    # ==============================

    st.subheader("Prediction Confidence")

    prob_df = pd.DataFrame({
        "Class": ["Normal", "Overweight", "Obese"],
        "Probability": probabilities
    })

    st.bar_chart(prob_df.set_index("Class"))

    # ==============================
    # INTERPRETATION
    # ==============================

    st.subheader("Clinical Interpretation")

    if prediction == 3:
        st.error("⚠️ High metabolic risk detected. Medical consultation recommended.")
    elif prediction == 2:
        st.warning("⚠️ Moderate risk. Lifestyle changes advised.")
    else:
        st.success("✅ Healthy metabolic profile.")

    # ==============================
    # FEATURE VALUES DISPLAY
    # ==============================

    st.subheader("Computed Indicators")

    st.write(f"HOMA-IR: {homa_ir:.2f}")
    st.write(f"TG/HDL Ratio: {tg_hdl:.2f}")