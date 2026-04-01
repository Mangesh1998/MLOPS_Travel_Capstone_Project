from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and feature columns
model = joblib.load('flight_price_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

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
    
    # Encode categoricals
    input_df = pd.get_dummies(input_df, columns=['from', 'to', 'flightType', 'agency'], drop_first=True)
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[feature_columns]
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    return jsonify({'predicted_price': prediction})

if __name__ == '__main__':
    app.run(debug=True)