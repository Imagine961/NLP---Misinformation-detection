from tqdm import tqdm
import pandas as pd
import string
import pytesseract
import pyautogui
from PIL import Image
from time import sleep
import tkinter as tk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os

# Check if model and vectorizer exist
model_path = "news_model.pkl"
vectorizer_path = "vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    print("Loading existing model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    # Loading Datasets
    print("Loading datasets...")
    fake_df = pd.read_csv("Fake.csv")
    true_df = pd.read_csv("True.csv")

    fake_df["label"] = 0
    true_df["label"] = 1

    # Combining and cleaning text data
    df = pd.concat([fake_df[["text", "label"]], true_df[["text", "label"]]], ignore_index=True)
    df = df.dropna(subset=['text', 'label'])

    # Parallel Text Cleaning
    def clean_text(text):
        if isinstance(text, str):
            text = text.lower().translate(str.maketrans('', '', string.punctuation))
            return text
        return ""

    print("Cleaning text data using parallel processing...")
    df['text'] = Parallel(n_jobs=-1)(delayed(clean_text)(t) for t in df['text'])

    # Splitting and vectorising data
    print("Splitting data and vectorising...")
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=1, max_features=40000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Training Optimised Naive Bayes Model
    print("Training Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Saving model and vectorizer
    print("Saving model and vectorizer...")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    # Evaluation with Comprehensive Metrics
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    print("Model Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Function to Display Confusion Matrix
def confusion_matrix_display():
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Prompt-based classification
def classify_text(text):
    cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))
    vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(vec)[0]
    return cleaned_text, prediction

while True:

    
    user_text = input("\nEnter a news article (or type 'exit' to quit): ")
    if user_text.lower() == 'exit':
        print("Exiting...")
        break

    text, label = classify_text(user_text)
    result = "REAL NEWS" if label == 1 else "FAKE NEWS"
    print(f"\nResult: {result}\n")
