import flask
from flask import Flask, render_template, request, jsonify
import joblib
import nltk
from nltk.corpus import stopwords
import numpy as np
import string

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('sentiment_analysis_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Download NLTK stopwords (if not already downloaded)
nltk.download('stopwords')

# Preprocessing function
def clean_text(text):
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Home route to render the HTML page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Endpoint to handle sentiment analysis requests
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get user input
        user_input = request.form.get('text')
        if not user_input:
            return jsonify({'error': 'No text provided'}), 400

        # Clean and preprocess the input
        cleaned_text = clean_text(user_input)
        # Vectorize the input
        text_vec = vectorizer.transform([cleaned_text])
        # Predict sentiment
        prediction = model.predict(text_vec)
        sentiment = prediction[0]

        # Return the sentiment as JSON
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        # Log the error and return a 500 response
        print(f"Error: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)