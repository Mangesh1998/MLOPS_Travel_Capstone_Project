import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load data
users_df = pd.read_csv('../data/users.csv')
hotels_df = pd.read_csv('../data/hotels.csv')
flights_df = pd.read_csv('../data/flights.csv')

# Merge for recommendations
# For simplicity, recommend hotels based on user's past bookings or similar users

# Let's create a simple content-based recommender based on hotel place and user age/gender

# But to make it ML, perhaps use KNN or something.

# For now, let's use cosine similarity on features.

# Features for hotels: place, price, days
hotels_df['features'] = hotels_df['place'] + ' ' + hotels_df['price'].astype(str) + ' ' + hotels_df['days'].astype(str)

# Vectorize
vectorizer = TfidfVectorizer()
hotel_vectors = vectorizer.fit_transform(hotels_df['features'])

# Save
joblib.dump(vectorizer, 'hotel_vectorizer.pkl')
joblib.dump(hotel_vectors, 'hotel_vectors.pkl')
hotel_data = hotels_df[['name', 'place', 'price', 'days']]
hotel_data.to_csv('hotel_data.csv', index=False)

# For recommendation, given a user, find similar hotels based on their past or profile.

# But for API, perhaps recommend based on input features.