import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
import numpy as np

# Set MLFlow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load all datasets
users_df = pd.read_csv('c:/Users/rohan/Desktop/travel_capstone_project/data/users.csv')
flights_df = pd.read_csv('c:/Users/rohan/Desktop/travel_capstone_project/data/flights.csv')
hotels_df = pd.read_csv('c:/Users/rohan/Desktop/travel_capstone_project/data/hotels.csv')

print(f"Users: {users_df.shape}, Flights: {flights_df.shape}, Hotels: {hotels_df.shape}")

# Feature Engineering for Users
users_df['name_length'] = users_df['name'].str.len()
users_df['name_words'] = users_df['name'].str.split().str.len()
users_df['first_name'] = users_df['name'].str.split().str[0]
users_df['last_name'] = users_df['name'].str.split().str[-1]
users_df['first_letter'] = users_df['name'].str[0].str.upper()
users_df['age_group'] = pd.cut(users_df['age'], bins=[0, 20, 30, 40, 50, 100], labels=['<20', '20-30', '30-40', '40-50', '50+'])

# Encode categorical features
le_first_letter = LabelEncoder()
le_first_name = LabelEncoder()
le_last_name = LabelEncoder()

users_df['first_letter_encoded'] = le_first_letter.fit_transform(users_df['first_letter'])
users_df['first_name_encoded'] = le_first_name.fit_transform(users_df['first_name'])
users_df['last_name_encoded'] = le_last_name.fit_transform(users_df['last_name'])
users_df['age_group_encoded'] = users_df['age_group'].cat.codes

# Company encoding
users_df = pd.get_dummies(users_df, columns=['company'], drop_first=True)

# Aggregate flight features per user
flight_features = flights_df.groupby('userCode').agg({
    'price': ['mean', 'std', 'count'],
    'distance': ['mean', 'std'],
    'time': ['mean', 'std'],
    'from': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown',
    'to': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown',
    'flightType': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown',
    'agency': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
}).reset_index()

# Flatten column names
flight_features.columns = ['userCode'] + [f'flight_{col[0]}_{col[1]}' if col[1] else f'flight_{col[0]}' for col in flight_features.columns[1:]]
flight_features = flight_features.rename(columns={
    'flight_from_<lambda>': 'flight_most_common_from',
    'flight_to_<lambda>': 'flight_most_common_to',
    'flight_flightType_<lambda>': 'flight_most_common_type',
    'flight_agency_<lambda>': 'flight_most_common_agency'
})

# Aggregate hotel features per user
hotel_features = hotels_df.groupby('userCode').agg({
    'price': ['mean', 'std', 'count'],
    'days': ['mean', 'std'],
    'total': ['mean', 'std'],
    'place': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown',
    'name': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
}).reset_index()

# Flatten column names
hotel_features.columns = ['userCode'] + [f'hotel_{col[0]}_{col[1]}' if col[1] else f'hotel_{col[0]}' for col in hotel_features.columns[1:]]
hotel_features = hotel_features.rename(columns={
    'hotel_place_<lambda>': 'hotel_most_common_place',
    'hotel_name_<lambda>': 'hotel_most_common_name'
})

# Merge all features
user_features = users_df.merge(flight_features, left_on='code', right_on='userCode', how='left')
user_features = user_features.merge(hotel_features, left_on='code', right_on='userCode', how='left')

# Fill missing values
user_features = user_features.fillna({
    'flight_price_mean': 0,
    'flight_price_std': 0,
    'flight_price_count': 0,
    'flight_distance_mean': 0,
    'flight_distance_std': 0,
    'flight_time_mean': 0,
    'flight_time_std': 0,
    'hotel_price_mean': 0,
    'hotel_price_std': 0,
    'hotel_price_count': 0,
    'hotel_days_mean': 0,
    'hotel_days_std': 0,
    'hotel_total_mean': 0,
    'hotel_total_std': 0
})

# Fill categorical missing values
user_features['flight_most_common_from'] = user_features['flight_most_common_from'].fillna('Unknown')
user_features['flight_most_common_to'] = user_features['flight_most_common_to'].fillna('Unknown')
user_features['flight_most_common_type'] = user_features['flight_most_common_type'].fillna('Unknown')
user_features['flight_most_common_agency'] = user_features['flight_most_common_agency'].fillna('Unknown')
user_features['hotel_most_common_place'] = user_features['hotel_most_common_place'].fillna('Unknown')
user_features['hotel_most_common_name'] = user_features['hotel_most_common_name'].fillna('Unknown')

# Encode additional categorical features
le_flight_from = LabelEncoder()
le_flight_to = LabelEncoder()
le_flight_type = LabelEncoder()
le_flight_agency = LabelEncoder()
le_hotel_place = LabelEncoder()
le_hotel_name = LabelEncoder()

user_features['flight_from_encoded'] = le_flight_from.fit_transform(user_features['flight_most_common_from'])
user_features['flight_to_encoded'] = le_flight_to.fit_transform(user_features['flight_most_common_to'])
user_features['flight_type_encoded'] = le_flight_type.fit_transform(user_features['flight_most_common_type'])
user_features['flight_agency_encoded'] = le_flight_agency.fit_transform(user_features['flight_most_common_agency'])
user_features['hotel_place_encoded'] = le_hotel_place.fit_transform(user_features['hotel_most_common_place'])
user_features['hotel_name_encoded'] = le_hotel_name.fit_transform(user_features['hotel_most_common_name'])

# Select features for model
feature_cols = [
    'age', 'name_length', 'name_words', 'first_letter_encoded', 'first_name_encoded', 'last_name_encoded', 'age_group_encoded',
    'flight_price_mean', 'flight_price_std', 'flight_price_count',
    'flight_distance_mean', 'flight_distance_std', 'flight_time_mean', 'flight_time_std',
    'flight_from_encoded', 'flight_to_encoded', 'flight_type_encoded', 'flight_agency_encoded',
    'hotel_price_mean', 'hotel_price_std', 'hotel_price_count',
    'hotel_days_mean', 'hotel_days_std', 'hotel_total_mean', 'hotel_total_std',
    'hotel_place_encoded', 'hotel_name_encoded'
] + [col for col in user_features.columns if col.startswith('company_')]

X = user_features[feature_cols]
y = user_features['gender']

print(f"Final feature matrix shape: {X.shape}")
print(f"Features used: {len(feature_cols)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Try different models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    with mlflow.start_run(run_name=f"Gender_Classifier_{name}"):
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f'{name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f'{name} Test Accuracy: {accuracy:.4f}')
        print(classification_report(y_test, y_pred))

        # Log metrics
        mlflow.log_metric('cv_accuracy_mean', cv_scores.mean())
        mlflow.log_metric('cv_accuracy_std', cv_scores.std())
        mlflow.log_metric('test_accuracy', accuracy)
        mlflow.sklearn.log_model(model, 'model')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

print(f'\nBest model: {best_model.__class__.__name__} with accuracy: {best_accuracy:.4f}')

# Save best model and preprocessors
joblib.dump(best_model, 'gender_model.pkl')
joblib.dump(scaler, 'gender_scaler.pkl')
joblib.dump(feature_cols, 'gender_features.pkl')

# Save encoders
encoders = {
    'le_first_letter': le_first_letter,
    'le_first_name': le_first_name,
    'le_last_name': le_last_name,
    'le_flight_from': le_flight_from,
    'le_flight_to': le_flight_to,
    'le_flight_type': le_flight_type,
    'le_flight_agency': le_flight_agency,
    'le_hotel_place': le_hotel_place,
    'le_hotel_name': le_hotel_name
}
joblib.dump(encoders, 'gender_encoders.pkl')