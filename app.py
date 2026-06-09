import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("car_price_model.pkl")

st.title("🚗 Car Price Prediction")

present_price = st.number_input(
    "Present Price (Lakhs)",
    min_value=0.0,
    value=0.0
)

driven_kms = st.number_input(
    "Driven KMs",
    min_value=0,
    value=0
)

owner = st.selectbox(
    "Number of Previous Owners",
    [0, 1, 2, 3]
)

car_age = st.number_input(
    "Car Age (Years)",
    min_value=0,
    value=0
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

selling_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

if st.button("Predict Price"):

    data = pd.DataFrame({
        "Present_Price": [present_price],
        "Driven_kms": [driven_kms],
        "Owner": [owner],
        "Car_Age": [car_age],
        "Fuel_Type": [fuel_type],
        "Selling_type": [selling_type],
        "Transmission": [transmission]
    })

    prediction = model.predict(data)

    st.success(
        f"Estimated Selling Price: ₹ {prediction[0]:.2f} Lakhs"
    )
