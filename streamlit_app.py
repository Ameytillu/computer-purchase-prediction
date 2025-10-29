import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("💻 Computer Purchase Prediction App")

# -----------------------------
# User inputs
# -----------------------------
st.header("Enter User Details")

age = st.slider("🎂 Age", 18, 70, 30)
income = st.number_input("💵 Area Income", min_value=10000.0, max_value=200000.0, value=60000.0)
internet_usage = st.slider("🌐 Daily Internet Usage (minutes)", 20, 300, 150)
gender = st.selectbox("🧍 Gender", ["Male", "Female"])

# New inputs for missing columns
ad_topic = st.text_input("📰 Ad Topic Line", "Example Ad Headline")
city = st.text_input("🏙️ City", "New York")
country = st.text_input("🌍 Country", "United States")

# -----------------------------
# Prepare input DataFrame
# -----------------------------
input_data = pd.DataFrame({
    'Daily Time Spent on Site': [0],   # Not collected, use 0 or similar dummy value
    'Age': [age],
    'Area Income': [income],
    'Daily Internet Usage': [internet_usage],
    'Ad Topic Line': [ad_topic],
    'City': [city],
    'Male': [1 if gender == "Male" else 0],
    'Country': [country]
})

# -----------------------------
# Prediction button
# -----------------------------
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"✅ Likely to Buy a Computer (Probability: {proba:.2f})")
        else:
            st.warning(f"❌ Not Likely to Buy a Computer (Probability: {proba:.2f})")

        # Simple visualization
        st.write("### Model Confidence")
        st.progress(float(proba))

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# Footer
st.write("---")
st.caption("Created by Amey Tillu | MS in Hospitality & Tourism Data Analytics | Streamlit ML App 🚀")

