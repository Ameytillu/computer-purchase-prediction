import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------------------------------------
# Load trained model (works both locally and on Streamlit Cloud)
# ---------------------------------------------------------
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ---------------------------------------------------------
# App Title and Description
# ---------------------------------------------------------
st.title("💻 Computer Purchase Prediction App")
st.write(
    "This app predicts whether a user is likely to buy a computer "
    "based on their online activity and demographic information."
)

# ---------------------------------------------------------
# User Inputs Section
# ---------------------------------------------------------
st.header("📋 Enter User Details")

daily_time = st.slider("🕒 Daily Time Spent on Site (minutes)", 20, 200, 60)
age = st.slider("🎂 Age", 10, 80, 30)
area_income = st.number_input("💵 Area Income", min_value=10000.0, max_value=200000.0, value=50000.0)
daily_internet_usage = st.slider("🌐 Daily Internet Usage (minutes)", 20, 300, 120)
gender = st.selectbox("🧍 Gender", ["Male", "Female"])
male = 1 if gender == "Male" else 0

# Prepare data for model
input_data = pd.DataFrame({
    "Daily Time Spent on Site": [daily_time],
    "Age": [age],
    "Area Income": [area_income],
    "Daily Internet Usage": [daily_internet_usage],
    "Male": [male]
})

# ---------------------------------------------------------
# Prediction Section
# ---------------------------------------------------------
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[0][1]

        st.subheader("🎯 Prediction Result")
        if prediction == 1:
            st.success(f"✅ Likely to Buy a Computer (Probability: {proba:.2f})")
        else:
            st.warning(f"❌ Not Likely to Buy a Computer (Probability: {proba:.2f})")

        # ---------------------------------------------------------
        # Visualization
        # ---------------------------------------------------------
        st.subheader("📊 Visualization")

        # Confidence progress bar
        st.write("**Model Confidence**")
        st.progress(float(proba))

        # Bar chart of user inputs
        st.write("**User Input Summary**")
        st.bar_chart(input_data.T)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.write("---")
st.caption("Created by Amey Tillu | MS in Hospitality & Tourism Data Analytics | Streamlit ML App 🚀")
