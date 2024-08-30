#!/usr/bin/env python3

import time
import pickle
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Constants
TEXT_COLUMN = 'tweet'
LABEL_COLUMN = 'label'
CLEANED_TEXT_COLUMN = 'cleaned_text'
VECTORIZER_SAVE_PATH = 'vectorizer.pkl'
SAVE_FILE_NAME = 'saved_model.pkl'

def load_data(file_path):
    print("Loading data...")
    data = pd.read_csv(file_path)
    print(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data

def clean_text(text):
    # print("Cleaning text...", text)
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
    # print("Text cleaned.", cleaned_text)
    return cleaned_text

def preprocess_data(data):
    print("Preprocessing data...")
    data['cleaned_text'] = data[TEXT_COLUMN].apply(clean_text)
    print("Data preprocessing completed.")
    return data

def vectorize_text(train_texts, test_texts):
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    print("Text data vectorized.")

    #save vectorizer for use in django
    with open(VECTORIZER_SAVE_PATH, 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)

    return X_train, X_test

def train_models(X_train, y_train):
    print("Training models...")
    models = {
        'Support Vector Machine': SVC(),
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': MultinomialNB()
    }
    trained_models = {}
    for name, model in models.items():
        print("Training",name)
        start_time = time.time()
        model.fit(X_train, y_train)
        trained_models[name] = model
        end_time = time.time()
        print(f"{name} model trained. Took {end_time - start_time} seconds.")
        print("------------------------------------------------------------------")
    return trained_models

def evaluate_models(models, X_test, y_test):
    print("Evaluating models...")
    i = 0
    for name, model in models.items():
        start_time = time.time()
        y_pred = model.predict(X_test)
        end_time = time.time()
        print(f"\n\nModel #{i}, {name} have been evaluated. Took {end_time - start_time} seconds.")
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{name} Model Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        i += 1
    print("Model evaluation completed.")
    print("------------------------------------------------------------------")

def save_and_evaluate_models(models, X_test, y_test):
    evaluate_models(models, X_test, y_test)

    print("Which models do you want to save?") #evaluate_models() should print out test data
    num_models = len(models)
    keys = list(models.keys())
    values = list(models.values())
    for i in range(0,num_models):
        print(f"{i}) {keys[i]}")
    choice = input("\nSelect no.>>> ")
    choice = int(choice)

    print(f"Saving model {keys[choice]}...{values[choice]}")
    with open(SAVE_FILE_NAME, 'wb') as model_file:
        pickle.dump(values[choice], model_file)

def load_saved_models():
    with open(SAVE_FILE_NAME, 'rb') as model_file:
        model = pickle.load(model_file)

    return model

def main():
    # Load data
    data = load_data('data/train.csv')  # Replace 'data/train.csv' with your dataset path
    # Preprocess data
    data = preprocess_data(data)
    # Split data into training and testing sets
    X = data[CLEANED_TEXT_COLUMN]
    y = data[LABEL_COLUMN]
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")
    # Vectorize text data
    X_train, X_test = vectorize_text(X_train_texts, X_test_texts)
    # Train models
    models = train_models(X_train, y_train)
    # Evaluate models
    # evaluate_models(models, X_test, y_test)
    save_and_evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()