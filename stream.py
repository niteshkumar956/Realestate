import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown

# Function to download the model files
@st.cache_resource
def load_model():
    url_model = "https://drive.google.com/uc?id=1sFJvTuBxvn62UIuNNu2Cy-pa6Seq6oNV"
    url_ohe = "https://drive.google.com/uc?id=1hvZAjZrKordRkzG-j-06qxGj16HW00dT"
    url_scaler = "https://drive.google.com/uc?id=16hluVlHWYDu0ALTPE3QWsuYvHdwLKJ2o"

    gdown.download(url_model, "house_price_model.pkl", quiet=False)
    gdown.download(url_ohe, "ohe_encoder.pkl", quiet=False)
    gdown.download(url_scaler, "scaler.pkl", quiet=False)

    model = joblib.load("house_price_model.pkl")
    ohe = joblib.load("ohe_encoder.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, ohe, scaler

# Load trained model and encoders
model, ohe, scaler = load_model()

# UI Components
st.title("🏡 House Price Prediction App")
st.write("Enter property details to get estimated price.")

area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, step=10)
bedroom = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
bathroom = st.selectbox("Bathrooms", [1, 2, 3, 4])
layout_type = st.selectbox("Layout Type", ["BHK", "RK"])
property_type = st.selectbox("Property Type", ["Apartment", "Independent House"])
furnish_type = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Ahmedabad"])

# Feature Engineering
log_area = np.log1p(area)
data = pd.DataFrame({
    'area': [area], 'bedroom': [bedroom], 'bathroom': [bathroom], 'log_area': [log_area],
    'layout_type': [layout_type], 'property_type': [property_type], 'furnish_type': [furnish_type], 'city': [city]
})

# Encode categorical features
data_encoded = pd.DataFrame(ohe.transform(data[['layout_type', 'property_type', 'furnish_type', 'city']]),
                            columns=ohe.get_feature_names_out())
data = data.drop(columns=['layout_type', 'property_type', 'furnish_type', 'city']).reset_index(drop=True)
data = pd.concat([data, data_encoded], axis=1)

# Scale features
data_scaled = scaler.transform(data)

# Predict price
if st.button("Predict Price"):
    price = model.predict(data_scaled)[0]
    st.success(f"Estimated House Price: ₹ {price:,.2f}") 
