import os
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Paths to the saved model and vectorizer
MODEL_PATH = 'saved_model.pkl'  # Update with the actual path
VECTORIZER_PATH = 'vectorizer.pkl'  # Update with the actual path

# Load the model
with open(MODEL_PATH, 'rb') as model_file:
    text_model = pickle.load(model_file)

# Load the vectorizer
with open(VECTORIZER_PATH, 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Preprocessing function
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    cleaned_text = ' '.join(words)
    return cleaned_text

def predict_text(text):
    # Preprocess the input text
    cleaned_text = preprocess_text(text)
    # Transform the text using the vectorizer
    transformed_text = vectorizer.transform([cleaned_text])
    # Predict using the model
    prediction = text_model.predict(transformed_text)[0]
    return prediction

def main():
    print("Enter text to check (or 'exit' to quit):")
    while True:
        user_input = input(">> ")
        if user_input.lower() == 'exit':
            break
        result = predict_text(user_input)
        if result == 1:
            print("Banned Text: This text violates the terms of use.")
        else:
            print("Allowed Text: This text is acceptable.")

if __name__ == "__main__":
    main()