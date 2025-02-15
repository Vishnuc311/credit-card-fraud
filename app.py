import streamlit as st
import pickle
import numpy as np

# Load trained model & scaler
with open("models/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to check if it's fraudulent.")

# Input fields
amount = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f")
time = st.number_input("Time (Seconds Since First Transaction)", min_value=0)

# Inputs for PCA features (V1 to V28)
features = []
for i in range(1, 29):
    value = st.number_input(f"V{i} Feature Value", format="%.4f")
    features.append(value)

# Predict button
if st.button("Predict Fraudulent Transaction"):
    # Prepare input array
    input_data = np.array([[time, amount] + features])
    input_data[:, :2] = scaler.transform(input_data[:, :2])  # Normalize time & amount

    # Get prediction & probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"🚨 Its Fraudulent Transaction Detected! (Probability: {probability:.4f})")
    else:
        st.success(f"✅ Its Legitimate Transaction (Probability: {probability:.4f})")
