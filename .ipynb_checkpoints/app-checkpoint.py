import streamlit as st
import pandas as pd
import numpy as np
import pickle

# FIX 2 & 3 & 4: Load model, scaler AND feature list together
with open('malaria_model.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
scaler = artifacts['scaler']
feature_cols = artifacts['feature_cols']
feature_medians = artifacts['feature_medians']

st.title("Malaria Incidence Prediction App — Zimbabwe")
st.markdown("Enter known malaria indicators to predict the estimated incidence rate (per 1,000 population at risk).")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", min_value=2000, max_value=2030, value=2025)
    malaria_mortality = st.number_input(
        "Malaria Mortality Rate",
        min_value=0.0, max_value=100.0, value=5.0, step=0.5,
        help="Estimated malaria mortality rate"
    )
    conf_cases = st.number_input(
        "Confirmed Malaria Cases",
        min_value=0, max_value=200000, value=5000, step=1000,
        help="Number of confirmed malaria cases"
    )

with col2:
    suspects = st.number_input(
        "Malaria Suspects",
        min_value=0, max_value=500000, value=10000, step=1000,
        help="Number of suspected malaria cases"
    )
    rdt_pos = st.number_input(
        "RDT Positive Tests",
        min_value=0, max_value=100000, value=3000, step=500,
        help="Number of Rapid Diagnostic Test positive results"
    )
    imported_cases = st.number_input(
        "Imported Malaria Cases",
        min_value=0, max_value=10000, value=100, step=50,
        help="Number of imported malaria cases"
    )

if st.button("Predict", type="primary"):
    # Build a feature row aligned to training columns
    # Start with medians (handles lag/cyclic columns not entered by user)
    row = dict(feature_medians)

    # Override with user inputs
    row['MALARIA_EST_MORTALITY']  = malaria_mortality
    row['MALARIA_CONF_CASES']     = conf_cases
    row['MALARIA_SUSPECTS']       = suspects
    row['MALARIA_RDT_POS']        = rdt_pos
    row['MALARIA_IMPORTED']       = imported_cases
    row['year_sin']               = np.sin(2 * np.pi * year / 4)
    row['year_cos']               = np.cos(2 * np.pi * year / 4)

    input_df = pd.DataFrame([row])[feature_cols]

    # FIX 4: Apply the same scaler used during training
    input_scaled = scaler.transform(input_df)

    # FIX 2: index [0] — prediction returns a 1-element array
    prediction = model.predict(input_scaled)[0]

    st.success(f"### Predicted Malaria Incidence Rate: **{prediction:,.2f}** per 1,000 population at risk")
    st.caption(f"Prediction for year {year} based on the provided indicators.")
