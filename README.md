# Mental Health Trend Detection from Social Media (Synthetic Data Version)

## Overview

This project focuses on detecting and analyzing **mental health trends from social media posts** using Natural Language Processing (NLP) and Machine Learning techniques. It simulates real-world social media behavior using a **realistic synthetic dataset of 10,000 posts**, enabling trend analysis without using sensitive personal data.

The system classifies posts into:
- **0 → Normal mental state**
- **1 → Mental distress indicators**

It also tracks how mental distress evolves over time to identify potential **mental health trend patterns**.

---

## Dataset Description

The dataset (`mental_health_social_media_dataset.csv`) contains **10,000 synthetic social media posts** with realistic structure and noise.

### Features:

| Column | Description |
|--------|-------------|
| post_id | Unique identifier for each post |
| text | Simulated social media post text |
| label | 0 = normal, 1 = mental distress |
| date | Timestamp of post (time-series format) |
| likes | Simulated engagement metric |
| shares | Simulated engagement metric |

### Dataset Characteristics:
- Realistic emotional language (stress, anxiety, happiness, etc.)
- Emojis and hashtags for authenticity
- Time-based distribution (120-day window)
- Engagement bias (normal posts receive higher engagement on average)

---

## Project Workflow

### 1. Data Generation
A realistic synthetic dataset is generated to simulate social media mental health expressions.

### 2. Text Preprocessing
- Lowercasing text
- Removing noise and special characters
- Cleaning URLs and symbols

### 3. Feature Extraction
- TF-IDF vectorization is used to convert text into numerical features

### 4. Model Training
A Logistic Regression model is trained to classify mental health states.

### 5. Evaluation
Model performance is evaluated using:
- Precision
- Recall
- F1-score
- Classification report

### 6. Trend Detection
Mental health trends are computed over time by aggregating predicted distress levels per day.

---

## Machine Learning Model

- **Algorithm:** Logistic Regression
- **Feature Engineering:** TF-IDF Vectorization
- **Task Type:** Binary Text Classification

---

## Key Insights

- The model identifies mental distress patterns from short text posts
- Time-based aggregation allows detection of **mental health trend fluctuations**
- Engagement metrics help simulate real-world social media behavior

---

## Visualization

The project generates a **time-series plot of mental distress rate**, showing how mental health conditions fluctuate across a 120-day period.

### Output:
- Mental Health Trend Over Time (Synthetic Data)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Regular Expressions (re)

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib
2. Run the script
python mental_health_trend.py
3. Output generated:
Model classification report
Trend analysis table
Time-series visualization plot
Project Structure
Mental-Health-Trend-Detection-from-Social-Media/
│
├── mental_health_social_media_dataset.csv
├── mental_health_trend.py
├── README.md
Future Improvements
Replace TF-IDF with BERT / transformer-based models
Add sentiment scoring instead of binary labels
Introduce real-world event simulation (exam stress, crises, holidays)
Deploy as a Streamlit dashboard
Integrate real-time social media APIs (Twitter/X, Reddit)
Ethical Note

This project uses fully synthetic data and does not involve real user data. It is designed strictly for:

Educational purposes
Research prototyping
Portfolio demonstration

No personal or sensitive data is used or stored.

## Author

Buka Jonah
