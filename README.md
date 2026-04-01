# MLOPS Travel Capstone Project

This project implements MLOps for travel data prediction and recommendation.

## Datasets
- users.csv: User information
- flights.csv: Flight data
- hotels.csv: Hotel bookings

## Models
1. Flight Price Regression Model
2. Gender Classification Model
3. Hotel Recommendation System

## Components
- Flask API for flight price prediction
- Streamlit app for recommendations
- Docker containerization
- Kubernetes deployment
- Airflow DAGs
- MLFlow tracking
- Jenkins CI/CD

## Setup
1. Install dependencies: pip install -r requirements.txt
2. Train models: python train_flight_model.py, etc.
3. Run API: python app.py
4. Run Streamlit: streamlit run recommendation_app.py
5. Build Docker: docker build -t flight-price-api .
6. Deploy to K8s: kubectl apply -f k8s_deployment.yaml

## Git
Initialized with main branch.