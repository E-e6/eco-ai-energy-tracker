import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load the trained model once
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Eco-AI Energy Tracker for Schools")
st.markdown("Track, predict and reduce your school’s energy use.")

# Input widgets for all 3 features
temperature = st.number_input('Temperature (°C)')
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0)
wind_speed = st.number_input('Wind Speed (km/h)', min_value=0.0)

if st.button('Predict Energy Consumption'):
    input_features = np.array([[temperature, humidity, wind_speed]])
    prediction = model.predict(input_features)
    st.write(f'Predicted Energy Consumption: {prediction[0]:.2f} kWh')

# Load and display data
df = pd.read_csv("energy_data.csv")
if st.checkbox("Show Energy Data"):
    st.dataframe(df)

# Plot energy usage
st.subheader("Energy Usage This Week")
fig, ax = plt.subplots()
ax.plot(df["Day"], df["Usage_kWh"], marker='o')
ax.set_ylabel("kWh Used")
ax.set_xlabel("Day")
st.pyplot(fig)

st.markdown("Built by [Eve Modra-O'Kane] for Young ICT Explorers 2025")