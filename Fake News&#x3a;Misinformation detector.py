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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("🔄 Loading datasets...")
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0
true_df["label"] = 1

if "text" not in fake_df.columns and "title" in fake_df.columns:
    fake_df["text"] = fake_df["title"] + " " + fake_df.get("text", "")
if "text" not in true_df.columns and "title" in true_df.columns:
    true_df["text"] = true_df["title"] + " " + true_df.get("text", "")

# Combining and cleaning text data
df = pd.concat([fake_df[["text", "label"]], true_df[["text", "label"]]], ignore_index=True)

# Cleaning function
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    return ""

print("🧹 Cleaning text data...")
df['text'] = [clean_text(t) for t in tqdm(df['text'])]
df = df.dropna(subset=['text', 'label'])

# Splitting and vectorising data
print("🔀 Splitting data and vectorising...")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_tfidf = vectorizer.fit_transform(tqdm(X_train, desc="Vectorising training set"))
X_test_tfidf = vectorizer.transform(X_test)

print("🧠 Training model (this may take a moment)...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("💾 Saving model and vectoriser...")
joblib.dump(model, "news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Evaluation
y_pred = model.predict(X_test_tfidf)
print("✅ Model Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# Prompt-based classification
def classify_text(text):
    cleaned_text = clean_text(text)
    vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(vec)[0]
    return cleaned_text, prediction

while True:
    user_text = input("\n📋 Enter a news article (or type 'exit' to quit): ")
    if user_text.lower() == 'exit':
        print("👋 Exiting...")
        break

    text, label = classify_text(user_text)
    result = "REAL NEWS" if label == 1 else "FAKE NEWS"
    print(f"\nResult: {result}\n")
