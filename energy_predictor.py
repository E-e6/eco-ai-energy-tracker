import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load the data
data = pd.read_csv("energy_data.csv")

# Prepare the data with 3 features
X = data[["Temperature_C", "Humidity", "Wind_Speed"]]
y = data["Usage_kWh"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained with 3 features and saved as model.pkl")