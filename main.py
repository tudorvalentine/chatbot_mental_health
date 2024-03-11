import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import random

# Load the intents data
with open('input/intents.json', 'r') as f:
    data = json.load(f)

# Convert data to DataFrame
df = pd.DataFrame(data['intents'])

# Split data into patterns and tags
X = df['patterns']
y = df['tag']
X_str = [' '.join(patterns) for patterns in X]
# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_str)

# Train a Support Vector Machine (SVM) classifier
model = SVC()
model.fit(X_vec, y)

# Function to predict intents based on user input
def predict_intent(user_input):
    user_input_vec = vectorizer.transform([user_input])
    intent = model.predict(user_input_vec)[0]
    return intent

# Function to generate responses based on predicted intents
def generate_response(intent):
    responses = df[df['tag'] == intent]['responses'].values[0]
    return random.choice(responses)

# Main function for the chatbot
def chat():
    print("Chatbot: Hello! How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye! Take care.")
            break

        # Predict intent
        intent = predict_intent(user_input)

        # Generate response
        response = generate_response(intent)

        print("Chatbot:", response)

if __name__ == "__main__":
    chat()
