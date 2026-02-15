import streamlit as st
import requests
import json

st.title("Wine Quality Prediction 🍷")

st.write("Adjust the features and click **Predict** to query the MLflow model API.")

# Feature inputs (match training columns)
fixed_acidity = st.slider("Fixed acidity", 4.0, 16.0, 7.0)
volatile_acidity = st.slider("Volatile acidity", 0.0, 1.5, 0.27)
citric_acid = st.slider("Citric acid", 0.0, 1.0, 0.36)
residual_sugar = st.slider("Residual sugar", 0.0, 20.0, 20.7)
chlorides = st.slider("Chlorides", 0.0, 0.2, 0.045)
free_sulfur_dioxide = st.slider("Free sulfur dioxide", 0.0, 100.0, 45.0)
total_sulfur_dioxide = st.slider("Total sulfur dioxide", 0.0, 300.0, 170.0)
density = st.slider("Density", 0.985, 1.01, 1.001)
ph = st.slider("pH", 2.5, 4.5, 3.0)
sulphates = st.slider("Sulphates", 0.2, 1.5, 0.45)
alcohol = st.slider("Alcohol", 8.0, 14.0, 8.8)

# Choose backend: local MLflow (port 5001)
api_url = "http://127.0.0.1:5001/invocations"

if st.button("Predict quality"):
    payload = {
        "dataframe_split": {
            "columns": [
                "fixed acidity","volatile acidity","citric acid",
                "residual sugar","chlorides","free sulfur dioxide",
                "total sulfur dioxide","density","pH",
                "sulphates","alcohol"
            ],
            "data": [[
                fixed_acidity, volatile_acidity, citric_acid,
                residual_sugar, chlorides, free_sulfur_dioxide,
                total_sulfur_dioxide, density, ph,
                sulphates, alcohol
            ]]
        }
    }

    try:
        resp = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10,
        )
        if resp.status_code == 200:
            pred = resp.json()["predictions"][0]
            st.success(f"Predicted quality: **{pred:.2f}**")
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
