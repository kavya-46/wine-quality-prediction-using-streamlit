import streamlit as st
from joblib import load
import numpy as np

# Load the trained model
model = load(r"D:\AI DS\final project AI ds\knn_wine_model.joblib")

# Streamlit app title
st.title("Wine Quality Prediction App")

import os
st.write(f"Current working directory: {os.getcwd()}")



# Description
st.write("""
This app predicts the **quality of wine** based on its chemical properties.
Please provide the input values below.
""")

# Input fields for wine features
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.4, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.7, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.076, step=0.001)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=500.0, value=34.0, step=1.0)
alcohol = st.number_input("Alcohol", min_value=0.0, max_value=20.0, value=9.4, step=0.1)
sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.56, step=0.01)

# Predict button
if st.button("Predict Wine Quality"):
    # Prepare the input features as a 2D array
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, chlorides, 
                          total_sulfur_dioxide, alcohol, sulphates]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Display the result
    st.write(f"### Predicted Wine Quality: {prediction[0]}")

# Footer
st.write("Developed with ❤️ using Streamlit")