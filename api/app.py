from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and preprocessors
model = joblib.load('../models/trained_models/flight_price_model.pkl')
scaler = joblib.load('../models/preprocessing/scaler.pkl')
le_from = joblib.load('../models/preprocessing/le_from.pkl')
le_to = joblib.load('../models/preprocessing/le_to.pkl')
le_agency = joblib.load('../models/preprocessing/le_agency.pkl')
features = joblib.load('../models/preprocessing/features.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Create a DataFrame from the input
    input_df = pd.DataFrame([data])
    
    # Preprocess similar to training
    input_df['date'] = pd.to_datetime(input_df['date'])
    input_df['year'] = input_df['date'].dt.year
    input_df['month'] = input_df['date'].dt.month
    input_df['day'] = input_df['date'].dt.day
    input_df['day_of_week'] = input_df['date'].dt.dayofweek
    
    # Encode categoricals
    input_df['from_encoded'] = le_from.transform(input_df['from'])
    input_df['to_encoded'] = le_to.transform(input_df['to'])
    input_df['agency_encoded'] = le_agency.transform(input_df['agency'])
    
    # Get dummies for flightType
    input_df = pd.get_dummies(input_df, columns=['flightType'], drop_first=True)
    
    # Feature engineering
    input_df['distance_per_time'] = input_df['distance'] / input_df['time']
    input_df['log_distance'] = np.log1p(input_df['distance'])
    input_df['log_time'] = np.log1p(input_df['time'])
    
    # Select features
    X = input_df[features]
    
    # Ensure all features are present
    for col in features:
        if col not in X.columns:
            X[col] = 0
    
    X = X[features]
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    
    return jsonify({'predicted_price': prediction})

if __name__ == '__main__':
    app.run(debug=True)