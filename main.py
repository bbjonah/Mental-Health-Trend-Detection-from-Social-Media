# Mental Health Trend Detection from Social Media (Synthetic Data Version)
# ---------------------------------------------------------------------
# This version generates a larger synthetic dataset to simulate social media posts

import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import random

# -----------------------------
# 1. SYNTHETIC DATA GENERATION
# -----------------------------
np.random.seed(42)
random.seed(42)

mental_distress_texts = [
    "I feel depressed and hopeless",
    "Anxiety is overwhelming me",
    "I am so lonely and sad",
    "I can't sleep and my mind is racing",
    "Feeling empty and exhausted",
    "Everything feels pointless",
]

normal_texts = [
    "I am feeling great today",
    "Life is good and I am happy",
    "Had a productive day",
    "Enjoying time with friends",
    "Feeling motivated and strong",
    "This is a wonderful day",
]

num_samples = 500
start_date = datetime(2026, 1, 1)

records = []

for i in range(num_samples):
    is_distress = np.random.choice([0, 1])
    if is_distress:
        text = random.choice(mental_distress_texts)
    else:
        text = random.choice(normal_texts)

    # Random date within 60 days
    date = start_date + timedelta(days=np.random.randint(0, 60))

    records.append({
        "text": text,
        "label": is_distress,
        "date": date
    })


df = pd.DataFrame(records)

# -----------------------------
# 2. TEXT PREPROCESSING
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


df['clean_text'] = df['text'].apply(clean_text)

# -----------------------------
# 3. FEATURE EXTRACTION
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# -----------------------------
# 4. TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. EVALUATION
# -----------------------------
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 6. TREND DETECTION
# -----------------------------
df['prediction'] = model.predict(X)

trend = df.groupby('date')['prediction'].mean().reset_index()
trend.rename(columns={'prediction': 'mental_distress_rate'}, inplace=True)

print("\nMental Health Trend Over Time:\n")
print(trend.head())

# -----------------------------
# 7. VISUALIZATION
# -----------------------------
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trend['date'], trend['mental_distress_rate'])
plt.title("Mental Health Trend Over Time (Synthetic Data)")
plt.xlabel("Date")
plt.ylabel("Distress Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# OPTIONAL IMPROVEMENTS
# -----------------------------
# - Increase dataset size (e.g., 10,000+ samples)
# - Add noise, slang, emojis for realism
# - Simulate spikes in distress (e.g., events)
# - Replace model with deep learning (BERT)
# - Deploy using Streamlit dashboard
