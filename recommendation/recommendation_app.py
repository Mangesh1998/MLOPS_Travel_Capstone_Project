import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load data
vectorizer = joblib.load('../models/hotel_vectorizer.pkl')
hotel_vectors = joblib.load('../models/hotel_vectors.pkl')
hotel_data = pd.read_csv('../models/hotel_data.csv')

st.title('Travel Recommendation System')

# Input for recommendation
place = st.selectbox('Preferred Place', hotel_data['place'].unique())
price = st.slider('Max Price per Day', 100, 500, 300)
days = st.slider('Number of Days', 1, 10, 3)

# Create feature string
query = f"{place} {price} {days}"
query_vector = vectorizer.transform([query])

# Compute similarity
similarities = cosine_similarity(query_vector, hotel_vectors).flatten()

# Get top recommendations
top_indices = similarities.argsort()[-5:][::-1]
recommendations = hotel_data.iloc[top_indices]

st.write('Recommended Hotels:')
st.dataframe(recommendations)