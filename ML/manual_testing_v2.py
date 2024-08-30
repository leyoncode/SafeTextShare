import os
import pickle
import re
import string
import argparse
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

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

def predict_text(text, model, vectorizer):
    # Preprocess the input text
    cleaned_text = preprocess_text(text)
    # Transform the text using the vectorizer
    transformed_text = vectorizer.transform([cleaned_text])
    # Predict using the model
    prediction = model.predict(transformed_text)[0]
    return prediction

def main(model_path, vectorizer_path):
    # Load the model
    with open(model_path, 'rb') as model_file:
        text_model = pickle.load(model_file)

    # Load the vectorizer
    with open(vectorizer_path, 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)

    print("Enter text to check (or 'exit' to quit):")
    while True:
        user_input = input(">> ")
        if user_input.lower() == 'exit':
            break
        result = predict_text(user_input, text_model, vectorizer)
        if result == 1:
            print("Banned Text: This text violates the terms of use.")
        else:
            print("Allowed Text: This text is acceptable.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text classification using a trained model.")
    parser.add_argument('--model', required=True, help='Path to the saved model file (e.g., saved_model.pkl)')
    parser.add_argument('--vectorizer', required=True, help='Path to the saved vectorizer file (e.g., vectorizer.pkl)')

    args = parser.parse_args()

    main(args.model, args.vectorizer)