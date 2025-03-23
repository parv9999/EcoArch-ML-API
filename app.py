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
