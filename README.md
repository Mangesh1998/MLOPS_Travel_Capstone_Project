# MLOPS Travel Capstone Project

This project implements MLOps for travel data prediction and recommendation.

## Project Structure
- `data/`: Datasets (users.csv, flights.csv, hotels.csv)
- `models/`: Model training scripts and saved models
- `api/`: Flask API for flight price prediction
- `recommendation/`: Streamlit app for hotel recommendations
- `airflow/`: Airflow DAGs for workflow orchestration
- `k8s/`: Kubernetes deployment files
- `docker/`: Dockerfiles and docker-compose
- `jenkins/`: CI/CD pipeline

## Models
1. **Flight Price Regression Model**
   - Algorithm: RandomForestRegressor
   - R² Score: 0.9868
   - CV Stability: ±0.0008
   - Features: Date, location, flight type, distance, time

2. **Gender Classification Model**
   - Algorithm: RandomForestClassifier (enhanced)
   - Accuracy: 37.7% (improved from 29.5%)
   - Features: Demographics + travel behavior (31 features)
   - CV Accuracy: 0.3433 ± 0.0734

3. **Hotel Recommendation System**
   - Algorithm: Content-based similarity
   - Features: Hotel location, price, days

## Setup
1. Install dependencies: pip install -r requirements.txt
2. Start MLFlow server: `python -m mlflow ui --host 0.0.0.0 --port 5000`
3. Train models: python models/train_flight_model.py etc.
4. Run all services: run_all.bat
5. Or individually:
   - API: python api/app.py
   - Streamlit: python -m streamlit run recommendation/recommendation_app.py
   - MLFlow: Already running at http://localhost:5000

## Docker
- Build API: docker build -f docker/Dockerfile -t flight-api .
- Run compose: docker-compose up

## Kubernetes
- Apply: kubectl apply -f k8s/

## Airflow
- Initialize Airflow and place DAG in dags folder

## Jenkins
- Use Jenkinsfile for CI/CD

## Git
Repository: https://github.com/Mangesh1998/MLOPS_Travel_Capstone_Project.git

## Git
Initialized with main branch.