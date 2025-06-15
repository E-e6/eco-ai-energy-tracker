import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Set page title
st.set_page_config(page_title="Eco-AI Energy Tracker", layout="centered")
st.title("Eco-AI Energy Tracker for Schools")
st.markdown("Track, predict and reduce your school’s energy use.")

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# --- INPUT SECTION ---
st.header("Predict Energy Consumption")

temperature = st.number_input('Temperature (°C)')
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0)
wind_speed = st.number_input('Wind Speed (km/h)', min_value=0.0)

if st.button('Predict Energy Consumption'):
    input_features = np.array([[temperature, humidity, wind_speed]])
    prediction = model.predict(input_features)
    st.success(f'Predicted Energy Consumption: **{prediction[0]:.2f} kWh**')

# --- LOAD AND SHOW DATA ---
df = pd.read_csv("energy_data.csv")

if st.checkbox("📊 Show Energy Data Table"):
    st.dataframe(df)

# --- DYNAMIC ENERGY USAGE PLOT ---
st.subheader("📈 Energy Usage This Week")

fig, ax = plt.subplots()

# Plot energy usage line
ax.plot(df["Day"], df["Usage_kWh"], marker='o', linestyle='-', color='green')

# Axis labels
ax.set_xlabel("Day")
ax.set_ylabel("kWh Used")
ax.set_title("Weekly Energy Usage")

# Dynamic Y-axis limits
y_min = df["Usage_kWh"].min()
y_max = df["Usage_kWh"].max()
y_range = y_max - y_min
ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

# Gridlines
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Show the plot
st.pyplot(fig)

# --- FOOTER ---
st.markdown("---")
st.markdown("Built by **Eve Modra-O'Kane** for **Young ICT Explorers 2025**")
