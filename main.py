import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import random

from telegram import Update
from telegram.ext import CommandHandler, filters, MessageHandler, ApplicationBuilder

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

# Function to handle the /start command
async def start(update, context):
    await update.message.reply_text("Hello! I'm your mental health bot. How can I help you today?")

# Function to handle incoming messages
async def handle_message(update, context):
    user_input = update.message.text
    intent = predict_intent(user_input)
    response = generate_response(intent)
    await update.message.reply_text(response)

def main():
    # Initialize the Updater and pass in your bot's token
    app = ApplicationBuilder().token("").build()

    # Register a command handler for the /start command
    app.add_handler(CommandHandler("start", start))

    # Register a message handler to handle incoming messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


    # Start the Bot
    app.run_polling()

    # Run the bot until you press Ctrl-C
    app.idle()

if __name__ == "__main__":
    main()
