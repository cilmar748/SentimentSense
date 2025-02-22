import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import string
import joblib

# Loading all the dataset
df = pd.read_csv('archive/Tweets.csv') 

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply text cleaning to the dataset
df['cleaned_text'] = df['text'].apply(clean_text)

# Split data into features and labels
X = df['cleaned_text']  # Features (cleaned text)
y = df['airline_sentiment']  # Labels (sentiment: positive, neutral, negative)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the model and vectorizer
joblib.dump(model, 'sentiment_analysis_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Interactive sentiment prediction
def predict_sentiment():
    while True:
        new_text = input('Please enter a text to check (or type "exit" to quit): ')
        
        if new_text.lower() == 'exit':
            print("Exiting the sentiment analysis tool. Goodbye!")
            break
        
        if not new_text.strip():
            print("Input cannot be empty. Please try again.")
            continue
        
        # Clean and predict
        cleaned_text = clean_text(new_text)
        text_vec = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vec)
        
        print(f'The text is (Sentiment): {prediction[0]}\n')

# Run the interactive system
predict_sentiment()