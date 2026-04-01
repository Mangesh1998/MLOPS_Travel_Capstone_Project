# MLOps Travel Capstone Project - Detailed Description

## Project Overview

This comprehensive MLOps project implements an end-to-end machine learning pipeline for travel and tourism data analysis. The system provides flight price prediction, user gender classification, and hotel recommendations through a fully containerized and orchestrated architecture.

## Business Context

In the competitive travel and tourism industry, accurate price prediction and personalized recommendations are crucial for enhancing customer experience and operational efficiency. This project leverages machine learning to:

- Predict flight prices with high accuracy
- Classify user demographics for targeted marketing
- Provide personalized hotel recommendations
- Demonstrate production-ready MLOps practices

## Dataset Description

### Users Dataset (users.csv)
- **Records**: 339 users
- **Features**:
  - `code`: Unique user identifier
  - `company`: Associated company (4You)
  - `name`: Full name of the user
  - `gender`: Gender classification (male/female/none)
  - `age`: Age of the user

### Flights Dataset (flights.csv)
- **Records**: 271,888 flight bookings
- **Features**:
  - `travelCode`: Travel booking identifier
  - `userCode`: User identifier (foreign key to users)
  - `from`: Origin city/state
  - `to`: Destination city/state
  - `flightType`: Type of flight (firstClass/economic)
  - `price`: Flight price (target variable)
  - `time`: Flight duration in hours
  - `distance`: Flight distance in km
  - `agency`: Booking agency
  - `date`: Booking date

### Hotels Dataset (hotels.csv)
- **Records**: 40,552 hotel bookings
- **Features**:
  - `travelCode`: Travel booking identifier
  - `userCode`: User identifier
  - `name`: Hotel name
  - `place`: Hotel location
  - `days`: Number of stay days
  - `price`: Price per day
  - `total`: Total booking price
  - `date`: Booking date

## Machine Learning Models

### 1. Flight Price Regression Model
**Algorithm**: RandomForestRegressor with regularization
**Performance Metrics**:
- R² Score: 0.9868 (98.68% variance explained)
- Cross-Validation R²: 0.9868 ± 0.0008
- Mean Squared Error: 1,782.18

**Features Used**:
- Temporal features: year, month, day, day_of_week
- Categorical encodings: from_encoded, to_encoded, agency_encoded
- Numerical features: time, distance
- Engineered features: distance_per_time, log_distance, log_time
- Flight type dummies

**Preprocessing**:
- Label encoding for high-cardinality categorical variables
- Standard scaling for numerical features
- Date feature extraction
- Noise injection for realistic performance

### 2. Gender Classification Model
**Algorithm**: RandomForestClassifier (improved with behavioral features)
**Performance Metrics**:
- Accuracy: 37.69% (improved from 29.48%)
- Cross-Validation Accuracy: 0.3433 ± 0.0734
- Precision/Recall/F1 by class:
  - Female: 0.47/0.42/0.45
  - Male: 0.36/0.41/0.38
  - None: 0.31/0.30/0.30

**Features Used (31 total)**:
- Demographic: age, age_group, name_length, name_words
- Name analysis: first_letter, first_name, last_name (encoded)
- Travel behavior: flight_price_mean, flight_count, most_common_destinations
- Hotel preferences: hotel_price_mean, hotel_count, preferred_places
- Company affiliation (one-hot encoded)

**Improvements Made**:
- Added behavioral features from flight and hotel booking history
- Name-based feature engineering
- Multiple algorithm comparison (RandomForest, GradientBoosting, LogisticRegression)
- Cross-validation for robust evaluation

### 3. Hotel Recommendation System
**Algorithm**: Content-based filtering with TF-IDF and Cosine Similarity
**Approach**:
- Feature vectorization: hotel place, price, days
- TF-IDF transformation
- Cosine similarity computation
- Top-k recommendation retrieval

## System Architecture

### Project Structure
```
travel_capstone_project/
├── data/                    # Raw datasets
│   ├── users.csv
│   ├── flights.csv
│   └── hotels.csv
├── models/                  # ML models and training
│   ├── train_flight_model.py
│   ├── train_gender_model.py
│   ├── train_recommendation_model.py
│   ├── flight_price_model.pkl
│   ├── gender_model.pkl
│   └── preprocessing artifacts
├── api/                     # Flask REST API
│   └── app.py
├── recommendation/          # Streamlit web app
│   └── recommendation_app.py
├── airflow/                 # Workflow orchestration
│   └── travel_dag.py
├── k8s/                     # Kubernetes deployment
│   └── k8s_deployment.yaml
├── docker/                  # Containerization
│   ├── Dockerfile
│   ├── Dockerfile.streamlit
│   └── docker-compose.yml
├── jenkins/                 # CI/CD pipeline
│   └── Jenkinsfile
└── requirements.txt         # Python dependencies
```

### Technology Stack

#### Machine Learning & Data Science
- **Python 3.9+**
- **scikit-learn**: ML algorithms and preprocessing
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **joblib**: Model serialization

#### MLOps & Experiment Tracking
- **MLFlow**: Model tracking and versioning
- **MLFlow UI**: Web interface for experiment management

#### Web Frameworks
- **Flask**: REST API for flight price prediction
- **Streamlit**: Interactive web app for recommendations

#### Orchestration & Automation
- **Apache Airflow**: Workflow scheduling and management
- **Jenkins**: Continuous Integration/Continuous Deployment

#### Containerization & Deployment
- **Docker**: Application containerization
- **Docker Compose**: Multi-service orchestration
- **Kubernetes**: Container orchestration and scaling

#### Version Control
- **Git**: Source code management
- **GitHub**: Remote repository hosting

## API Endpoints

### Flight Price Prediction API
**Endpoint**: `POST /predict`
**Input Format**:
```json
{
  "from": "Recife (PE)",
  "to": "Florianopolis (SC)",
  "flightType": "firstClass",
  "time": 1.76,
  "distance": 676.53,
  "agency": "FlyingDrops",
  "date": "09/26/2019"
}
```
**Output Format**:
```json
{
  "predicted_price": 1434.38
}
```

## Deployment Instructions

### Local Development Setup

1. **Environment Setup**:
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Start MLFlow Server**:
   ```bash
   python -m mlflow ui --host 0.0.0.0 --port 5000
   ```

3. **Train Models**:
   ```bash
   python models/train_flight_model.py
   python models/train_gender_model.py
   python models/train_recommendation_model.py
   ```

4. **Run Services**:
   ```bash
   # Flask API
   python api/app.py

   # Streamlit App
   python -m streamlit run recommendation/recommendation_app.py

   # MLFlow UI (already running)
   ```

### Docker Deployment

1. **Build Images**:
   ```bash
   docker build -f docker/Dockerfile -t flight-api .
   docker build -f docker/Dockerfile.streamlit -t recommendation-app .
   ```

2. **Run with Docker Compose**:
   ```bash
   docker-compose up
   ```

### Kubernetes Deployment

1. **Apply Manifests**:
   ```bash
   kubectl apply -f k8s/k8s_deployment.yaml
   ```

2. **Check Status**:
   ```bash
   kubectl get pods
   kubectl get services
   ```

## CI/CD Pipeline

### Jenkins Pipeline Stages
1. **Build**: Install dependencies and build Docker images
2. **Test**: Run unit tests and model validation
3. **Deploy**: Deploy to Kubernetes cluster

### Airflow DAG
- **Schedule**: Daily execution
- **Tasks**:
  - Data loading and validation
  - Model retraining
  - Performance monitoring

## Model Monitoring & Maintenance

### Performance Tracking
- MLFlow experiment tracking
- Cross-validation metrics
- Test set performance monitoring

### Model Retraining
- Automated retraining on new data
- Performance threshold monitoring
- Model versioning and rollback

## Future Enhancements

### Model Improvements
- **Flight Price Model**: Add external factors (fuel prices, demand, holidays)
- **Gender Classification**: Incorporate name analysis, behavioral data
- **Recommendations**: Implement collaborative filtering, user preferences

### System Enhancements
- **Real-time Inference**: Optimize for low-latency predictions
- **A/B Testing**: Implement model comparison framework
- **Feature Store**: Centralized feature management
- **Model Registry**: Production model versioning

### Scalability
- **Microservices Architecture**: Break down monolithic components
- **Database Integration**: Replace CSV files with proper database
- **Caching Layer**: Implement Redis for faster responses
- **Load Balancing**: Distribute traffic across multiple instances

## Performance Benchmarks

### System Performance
- **API Response Time**: <100ms for single predictions
- **Model Training Time**: ~30 seconds for flight model
- **Recommendation Retrieval**: <50ms for top-5 results

### Accuracy Metrics
- **Flight Price Prediction**: 98.7% R² score
- **Gender Classification**: 29.5% accuracy (baseline)
- **Hotel Recommendations**: Qualitative evaluation (user satisfaction)

## Security Considerations

### Data Privacy
- No sensitive user data stored
- Anonymized user identifiers
- GDPR compliance considerations

### Model Security
- Secure model artifact storage
- Input validation and sanitization
- Rate limiting on API endpoints

## Conclusion

This MLOps project demonstrates a complete machine learning lifecycle from data ingestion to production deployment. The system successfully implements:

- High-accuracy flight price prediction
- Scalable web services architecture
- Comprehensive experiment tracking
- Automated deployment pipelines
- Container orchestration

The project serves as a robust foundation for production ML systems in the travel industry, with clear pathways for future enhancements and scaling.

## Contact Information

For questions or contributions, please refer to the GitHub repository:
https://github.com/Mangesh1998/MLOPS_Travel_Capstone_Project

---

*Last Updated: April 1, 2026*
*Version: 1.0*