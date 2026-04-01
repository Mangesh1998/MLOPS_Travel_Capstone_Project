import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
import numpy as np

# Set MLFlow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load the data
print("Loading data...")
flights_df = pd.read_csv('c:/Users/rohan/Desktop/travel_capstone_project/data/flights.csv')
print(f"Data loaded: {flights_df.shape}")

# For testing, sample
flights_df = flights_df.sample(10000, random_state=42)
print(f"Sampled to: {flights_df.shape}")

# Add noise to price to make it realistic
np.random.seed(42)
noise = np.random.normal(0, flights_df['price'].std() * 0.1, len(flights_df))
flights_df['price'] += noise
flights_df['price'] = flights_df['price'].clip(lower=0)  # Ensure non-negative

# Preprocessing
# Convert date to datetime
flights_df['date'] = pd.to_datetime(flights_df['date'])

# Extract features from date
flights_df['year'] = flights_df['date'].dt.year
flights_df['month'] = flights_df['date'].dt.month
flights_df['day'] = flights_df['date'].dt.day
flights_df['day_of_week'] = flights_df['date'].dt.dayofweek

# Handle categorical variables with high cardinality using label encoding
le_from = LabelEncoder()
le_to = LabelEncoder()
le_agency = LabelEncoder()

flights_df['from_encoded'] = le_from.fit_transform(flights_df['from'])
flights_df['to_encoded'] = le_to.fit_transform(flights_df['to'])
flights_df['agency_encoded'] = le_agency.fit_transform(flights_df['agency'])

# Encode flightType (low cardinality)
flights_df = pd.get_dummies(flights_df, columns=['flightType'], drop_first=True)

# Feature engineering
flights_df['distance_per_time'] = flights_df['distance'] / flights_df['time']
flights_df['log_distance'] = np.log1p(flights_df['distance'])
flights_df['log_time'] = np.log1p(flights_df['time'])

# Features and target
features = ['year', 'month', 'day', 'day_of_week', 'from_encoded', 'to_encoded', 'agency_encoded', 'time', 'distance', 'distance_per_time', 'log_distance', 'log_time'] + [col for col in flights_df.columns if col.startswith('flightType_')]
X = flights_df[features]
y = flights_df['price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Ridge': Ridge(alpha=1.0)
}

best_model = None
best_r2 = -np.inf

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        print(f'{name} CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f'{name} Test MSE: {mse:.4f}, R2: {r2:.4f}')
        
        # Log
        mlflow.log_metric('cv_r2_mean', cv_scores.mean())
        mlflow.log_metric('cv_r2_std', cv_scores.std())
        mlflow.log_metric('test_mse', mse)
        mlflow.log_metric('test_r2', r2)
        mlflow.sklearn.log_model(model, 'model')
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model

print(f'Best model R2: {best_r2:.4f}')

# Save best model and preprocessors
joblib.dump(best_model, '../trained_models/flight_price_model.pkl')
joblib.dump(scaler, '../preprocessing/scaler.pkl')
joblib.dump(le_from, '../preprocessing/le_from.pkl')
joblib.dump(le_to, '../preprocessing/le_to.pkl')
joblib.dump(le_agency, '../preprocessing/le_agency.pkl')
joblib.dump(features, '../preprocessing/features.pkl')