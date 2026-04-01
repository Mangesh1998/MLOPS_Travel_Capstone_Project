import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn

# Set MLFlow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load the data
users_df = pd.read_csv('c:/Users/rohan/Desktop/travel_capstone_project/data/users.csv')

# Preprocessing
# Features: company, name, age (name might have info, but for simplicity, use company and age)
# Encode company
users_df = pd.get_dummies(users_df, columns=['company'], drop_first=True)

# Features and target
X = users_df.drop(['code', 'name', 'gender'], axis=1)
y = users_df['gender']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
with mlflow.start_run(run_name="Gender_Classifier"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))
    
    # Log metrics
    mlflow.log_metric('accuracy', accuracy)
    mlflow.sklearn.log_model(model, 'model')

# Save
joblib.dump(model, 'gender_model.pkl')
joblib.dump(X.columns.tolist(), 'gender_features.pkl')