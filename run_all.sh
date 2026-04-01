#!/bin/bash

# Run all training scripts
echo "Training models..."
python models/train_flight_model.py
python models/train_gender_model.py
python models/train_recommendation_model.py

# Start services
echo "Starting Flask API..."
python api/app.py &
echo "Starting Streamlit..."
python -m streamlit run recommendation/recommendation_app.py &
echo "Starting MLFlow..."
python -m mlflow ui &

echo "All services started. Access:"
echo "Flask API: http://localhost:5000"
echo "Streamlit: http://localhost:8501"
echo "MLFlow: http://localhost:5000"  # Default MLFlow port

wait