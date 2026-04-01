import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn

# Load the data
flights_df = pd.read_csv('flights.csv')

# Preprocessing
# Convert date to datetime
flights_df['date'] = pd.to_datetime(flights_df['date'])

# Extract features from date
flights_df['year'] = flights_df['date'].dt.year
flights_df['month'] = flights_df['date'].dt.month
flights_df['day'] = flights_df['date'].dt.day

# Encode categorical variables
flights_df = pd.get_dummies(flights_df, columns=['from', 'to', 'flightType', 'agency'], drop_first=True)

# Features and target
X = flights_df.drop(['travelCode', 'userCode', 'price', 'date'], axis=1)
y = flights_df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'MSE: {mse}')
    print(f'R2: {r2}')
    
    # Log metrics
    mlflow.log_metric('mse', mse)
    mlflow.log_metric('r2', r2)
    
    # Log model
    mlflow.sklearn.log_model(model, 'model')

# Save the model
joblib.dump(model, 'flight_price_model.pkl')

# Also save the feature columns for later use in API
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')