#!/usr/bin/env python3

#patch for intel cpu to boost training times
from sklearnex import patch_sklearn
patch_sklearn()

import time
from datetime import datetime
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
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Constants
DATA_FILENAME = 'data/final_dataset.csv'
TEXT_COLUMN = 'Content'
LABEL_COLUMN = 'Label'
CLEANED_TEXT_COLUMN = 'cleaned_text'
VECTORIZER_SAVE_PATH = 'vectorizer.pkl'
MODEL_SAVE_FILE_NAME = 'saved_model'


def print_log(message, default_log_file = 'log.txt'):
    current_time = datetime.now()

    log_entry = f"{current_time} ---- {message}"

    print(log_entry)

    # Write the log entry to the specified log file
    with open(default_log_file, 'a') as log_file:
        log_file.write(log_entry + '\n')


def load_data(file_path):
    print_log("Loading data...")
    data = pd.read_csv(file_path)
    print_log(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data

def clean_text(text):
    # print_log("Cleaning text...", text)
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
    # print_log("Text cleaned.", cleaned_text)
    return cleaned_text

def preprocess_data(data):
    print_log("Preprocessing data...")
    data['cleaned_text'] = data[TEXT_COLUMN].apply(clean_text)
    print_log("Data preprocessing completed.")
    return data

def vectorize_text(train_texts, test_texts):
    print_log("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    print_log("Text data vectorized.")

    #save vectorizer for use in django
    with open(VECTORIZER_SAVE_PATH, 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)

    return X_train, X_test

def train_models(X_train, y_train):
    print_log("Training models...")
    models = {
        # 'Support Vector Machine': SVC(),
        # 'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=50000),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': MultinomialNB(),
        'XGBoost': XGBClassifier(
            objective='binary:logistic',
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    }
    trained_models = {}
    for name, model in models.items():
        print_log("Training " + name)
        start_time = time.time()
        model.fit(X_train, y_train)
        trained_models[name] = model
        end_time = time.time()
        print_log(f"{name} model trained. Took {end_time - start_time} seconds.")
        print_log("------------------------------------------------------------------")
    return trained_models

def evaluate_models(models, X_test, y_test):
    print_log("Evaluating models...")
    i = 0
    for name, model in models.items():
        start_time = time.time()
        y_pred = model.predict(X_test)
        end_time = time.time()
        print_log(f"\n\nModel #{i}, {name} have been evaluated. Took {end_time - start_time} seconds.")
        accuracy = accuracy_score(y_test, y_pred)
        print_log(f"\n{name} Model Accuracy: {accuracy:.4f}")
        print_log("Classification Report:")
        print_log(classification_report(y_test, y_pred))
        print_log("Confusion Matrix:")
        print_log(confusion_matrix(y_test, y_pred))
        i += 1
    print_log("Model evaluation completed.")
    print_log("------------------------------------------------------------------")

def save_and_evaluate_models(models, X_test, y_test):
    evaluate_models(models, X_test, y_test)

    # print_log("Which models do you want to save?") #evaluate_models() should print out test data
    # num_models = len(models)
    # keys = list(models.keys())
    # values = list(models.values())
    # for i in range(0,num_models):
    #     print_log(f"{i}) {keys[i]}")
    # choice = input("\nSelect no.>>> ")
    # choice = int(choice)
    #
    # print_log(f"Saving model {keys[choice]}...{values[choice]}")
    # with open(MODEL_SAVE_FILE_NAME, 'wb') as model_file:
    #     pickle.dump(values[choice], model_file)

    for name, model in models.items():
        filename = f"{MODEL_SAVE_FILE_NAME}_{name}.pkl"
        print_log("Saving model to " + filename)

        with open(filename, 'wb') as model_file:
            pickle.dump(model, model_file)

def load_saved_models():
    with open(MODEL_SAVE_FILE_NAME, 'rb') as model_file:
        model = pickle.load(model_file)

    return model

def main():
    # Load data
    data = load_data(DATA_FILENAME)  # Replace 'data/train.csv' with your dataset path
    # Preprocess data
    data = preprocess_data(data)
    # Split data into training and testing sets
    X = data[CLEANED_TEXT_COLUMN]
    y = data[LABEL_COLUMN]
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print_log("Data have been split into training and testing sets.")
    # Vectorize text data
    X_train, X_test = vectorize_text(X_train_texts, X_test_texts)
    # Train models
    models = train_models(X_train, y_train)
    # Evaluate models
    # evaluate_models(models, X_test, y_test)
    save_and_evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()