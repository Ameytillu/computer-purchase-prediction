import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Path to your saved model
model_path = r"D:\Git Hub IMP\computer_purchase_prediction\model.pkl"

# Load the trained model
with open(model_path, "rb") as file:
    model = pickle.load(file)

st.title("💻 Computer Purchase Prediction App")
st.write("This app predicts whether a user is likely to buy a computer based on their online activity and demographics.")

# --- User Inputs ---
daily_time = st.slider("🕒 Daily Time Spent on Site (minutes)", 20, 200, 60)
age = st.slider("🎂 Age", 10, 80, 30)
area_income = st.number_input("💵 Area Income", min_value=10000.0, max_value=200000.0, value=50000.0)
daily_internet_usage = st.slider("🌐 Daily Internet Usage (minutes)", 20, 300, 120)
gender = st.selectbox("🧍 Gender", ["Male", "Female"])
male = 1 if gender == "Male" else 0

# Create dataframe for prediction
input_data = pd.DataFrame({
    'Daily Time Spent on Site': [daily_time],
    'Age': [age],
    'Area Income': [area_income],
    'Daily Internet Usage': [daily_internet_usage],
    'Male': [male]
})

if st.button("🔍 Predict"):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Likely to Buy a Computer (Probability: {proba:.2f})")
    else:
        st.warning(f"❌ Not Likely to Buy a Computer (Probability: {proba:.2f})")

# Add a simple visualization
st.write("### Visualization")
st.progress(float(proba))
