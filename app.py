import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

st.set_page_config(page_title="Eco-AI Energy Tracker", layout="centered")
st.title("Eco-AI Energy Tracker for Schools")
st.markdown("Track, predict and reduce your school’s energy use.")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load original energy data
df_path = "energy_data.csv"
if os.path.exists(df_path):
    df = pd.read_csv(df_path)
else:
    st.error("⚠️ Missing 'energy_data.csv'. Please upload it.")
    st.stop()

# --- INPUT SECTION ---
st.header("Predict Energy Consumption")

temperature = st.number_input('Temperature (°C)')
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0)
wind_speed = st.number_input('Wind Speed (km/h)', min_value=0.0)

if st.button('Predict Energy Consumption'):
    input_features = np.array([[temperature, humidity, wind_speed]])
    prediction = model.predict(input_features)
    st.success(f'Predicted Energy Consumption: **{prediction[0]:.2f} kWh**')

# --- EDITABLE DATA TABLE ---
st.subheader("📋 Edit Energy Data Table")
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# Save edited data to CSV (optional — you can also keep this in memory only)
edited_df.to_csv(df_path, index=False)

# --- UPDATED DYNAMIC PLOT ---
st.subheader("📈 Updated Energy Usage This Week")

fig, ax = plt.subplots()
ax.plot(edited_df["Day"], edited_df["Usage_kWh"], marker='o', linestyle='-', color='green')

# Axis labels and grid
ax.set_xlabel("Day")
ax.set_ylabel("kWh Used")
ax.set_title("Weekly Energy Usage")
ax.grid(True, linestyle='--', alpha=0.7)

# Dynamic Y range
y_min = edited_df["Usage_kWh"].min()
y_max = edited_df["Usage_kWh"].max()
y_range = y_max - y_min
ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

st.pyplot(fig)

# --- FOOTER ---
st.markdown("---")
st.markdown("Built by **Eve Modra-O'Kane** for **Young ICT Explorers 2025**")
