import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the data
users_df = pd.read_csv('users.csv')

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
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save
joblib.dump(model, 'gender_model.pkl')
joblib.dump(X.columns.tolist(), 'gender_features.pkl')