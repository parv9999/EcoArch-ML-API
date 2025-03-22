# ✅ Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from google.colab import files
import os
import joblib  # For saving and loading models


# ✅ Step 1: Check if the file exists, else upload manually

file_path = '/mnt/data/Updated_GreenRoofData.csv'  # Use a cleaner file name
if not os.path.exists(file_path):
    uploaded = files.upload()
    file_path = list(uploaded.keys())[0]


# ✅ Step 2: Load dataset safely

df = pd.read_csv(file_path, encoding='ISO-8859-1')


# ✅ Step 3: Display dataset info

print("Dataset Info:")   # SHOWS ALL 17 COLUMNS WITH 
df.info()                # ENTERIES AND THERE DATA TYPE


# ✅ Step 4: Handle missing values

df.fillna(df.select_dtypes(include=[np.number]).median(), inplace=True)


# ✅ Step 5: Fix Column Names (Removing Extra Spaces)

df.columns = df.columns.str.strip()


# ✅ Step 6: Define Features (X) and Target Variables (y)

X = df[['roof_area (sq.m)', 'avg_temperature (°C)', 'rainfall (mm/year)',
        'humidity (%)', 'sunlight_exposure (hours/day)', 'wind_speed (m/s)',
        'soil_depth (cm)', 'installation_cost (  ? )', 'maintenance_cost (   ? /year)']]


# ✅ Step 7: FIX  Encode Green Roof Type for Model Training 

roof_type_mapping = {name: idx for idx, name in enumerate(df['green_roof_type'].unique())}
df['green_roof_type'] = df['green_roof_type'].map(roof_type_mapping)   

# ✅ Fix: Encode categorical variable for Green Roof Type
#    df['green_roof_type'] = df['green_roof_type'].astype('category').cat.codes


# 🎯 **Step 8: Train Model for Cooling Effect**

y_cooling = df['cooling_effect (°C reduction)']
X_train, X_test, y_train, y_test = train_test_split(X, y_cooling, test_size=0.2, random_state=42)
cooling_model = RandomForestRegressor(n_estimators=100, random_state=42)
cooling_model.fit(X_train, y_train)

y_pred = cooling_model.predict(X_test)
print("Cooling Effect MAE:", mean_absolute_error(y_test, y_pred))


# 🎯 **Step 9: Train Model for Energy Savings**

y_energy = df['energy_savings (kWh/year)']
X_train, X_test, y_train, y_test = train_test_split(X, y_energy, test_size=0.2, random_state=42)
energy_model = RandomForestRegressor(n_estimators=100, random_state=42)
energy_model.fit(X_train, y_train)

y_pred = energy_model.predict(X_test)
print("Energy Savings MAE:", mean_absolute_error(y_test, y_pred))


# 🎯 **Step 10: Train Model for Green Roof Type**

y_roof_type = df['green_roof_type']
X_train, X_test, y_train, y_test = train_test_split(X, y_roof_type, test_size=0.2, random_state=42)
roof_type_model = KNeighborsClassifier(n_neighbors=5)
roof_type_model.fit(X_train, y_train)

y_pred = roof_type_model.predict(X_test)
print("Roof Type Prediction Accuracy:", accuracy_score(y_test, y_pred))


# 🎯 **Step 11: Train Model for Carbon Sequestration**

y_carbon = df['carbon_sequestration (kg/year)']
X_train, X_test, y_train, y_test = train_test_split(X, y_carbon, test_size=0.2, random_state=42)
carbon_model = RandomForestRegressor(n_estimators=100, random_state=42)
carbon_model.fit(X_train, y_train)

y_pred = carbon_model.predict(X_test)
print("Carbon Sequestration MAE:", mean_absolute_error(y_test, y_pred))


# ✅ Step 12: Save the Trained Models

joblib.dump(cooling_model, "cooling_model.pkl")
joblib.dump(energy_model, "energy_model.pkl")
joblib.dump(roof_type_model, "roof_type_model.pkl")
joblib.dump(carbon_model, "carbon_model.pkl")
joblib.dump(roof_type_mapping, "roof_type_mapping.pkl")
print("✅ All models saved successfully!")


# ✅ Step 13: Load the Saved Models for Prediction

cooling_model = joblib.load("cooling_model.pkl")
energy_model = joblib.load("energy_model.pkl")
roof_type_model = joblib.load("roof_type_model.pkl")
carbon_model = joblib.load("carbon_model.pkl")
roof_type_mapping = joblib.load("roof_type_mapping.pkl")
print("✅ Models loaded successfully!")


# Download to local system
from google.colab import files
files.download("cooling_model.pkl")
files.download("energy_model.pkl")
files.download("roof_type_model.pkl")
files.download("carbon_model.pkl")
files.download("roof_type_mapping.pkl")


# ✅ Step 14: New Data for Prediction (Ensure Correct Column Names & Types)

new_data = pd.DataFrame([[100 , 30.5 , 1200 , 75 , 6.5 ,  3.2 ,  15 , 50000 , 2000 ]], columns=X.columns)

# 🎯 **Step 15: Make Predictions**

effect = cooling_model.predict(new_data)
energy = energy_model.predict(new_data)
predicted_roof_type_idx = roof_type_model.predict(new_data)[0]
predicted_roof_type = list(roof_type_mapping.keys())[list(roof_type_mapping.values()).index(predicted_roof_type_idx)]
carbon = carbon_model.predict(new_data)


# 📌 Step 16: Print Predictions

print("\n📊 **Predictions for New Data**:")
print("Predicted Cooling Effect:", effect[0], "°C reduction")
print("Predicted Energy Savings:", energy[0], "kWh/year")
print("Recommended Green Roof Type:", predicted_roof_type)  # Mapped to category name
print("Predicted Carbon Sequestration:", carbon[0], "kg/year")





from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib  # For saving and loading models

# ✅ Load the trained models
cooling_model = joblib.load("cooling_model.pkl")
energy_model = joblib.load("energy_model.pkl")
roof_type_model = joblib.load("roof_type_model.pkl")
carbon_model = joblib.load("carbon_model.pkl")

# ✅ Load the roof type mapping
roof_type_mapping = joblib.load("roof_type_mapping.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return "Green Roof Prediction API is running!"

# ✅ Define the API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON input from user
    
    # Convert input to DataFrame
    input_data = pd.DataFrame([data])

    # Get predictions
    effect = cooling_model.predict(input_data)[0]
    energy = energy_model.predict(input_data)[0]
    predicted_roof_type_idx = roof_type_model.predict(input_data)[0]
    predicted_roof_type = list(roof_type_mapping.keys())[list(roof_type_mapping.values()).index(predicted_roof_type_idx)]
    carbon = carbon_model.predict(input_data)[0]

    # Return results as JSON
    response = {
        "Predicted Cooling Effect": f"{effect:.2f} °C reduction",
        "Predicted Energy Savings": f"{energy:.2f} kWh/year",
        "Recommended Green Roof Type": predicted_roof_type,
        "Predicted Carbon Sequestration": f"{carbon:.2f} kg/year"
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)







####-----------------------------------------------------------------------####

#cat.codes
#Assigns a unique numerical code to each category.

#Example conversion:

#Intensive Green Roof   → 0  
#Extensive Green Roof   → 1  
#Semi-Intensive Green Roof  → 2 

#🔹 Why is this used?
#1.Machine learning models cannot work with categorical text values directly.
#2.Encoding converts these text values into numerical form so they can be used 
#as input features.


#--------------------------------------------------------------------------####

#                               {{  either use those 2 line below all traing
# ✅ Step 12: Evaluate Models    all training modal or these two step for
#                                 MAE %accuracy calucalation   }}

#cooling_mae = mean_absolute_error(y_test, cooling_model.predict(X_test))
#energy_mae = mean_absolute_error(y_test, energy_model.predict(X_test))
#carbon_mae = mean_absolute_error(y_test, carbon_model.predict(X_test))
#roof_type_accuracy = accuracy_score(y_test, roof_type_model.predict(X_test))

# 📌 Step 16: Print Model Evaluation Metrics
#print("\n📊 **Model Evaluation Metrics**:")
#print(f"Cooling Effect MAE: {cooling_mae:.4f}")
#print(f"Energy Savings MAE: {energy_mae:.4f}")
#print(f"Roof Type Prediction Accuracy: {roof_type_accuracy:.4f}")
#print(f"Carbon Sequestration MAE: {carbon_mae:.4f}")

#---------------------------------------------------------------------------###





