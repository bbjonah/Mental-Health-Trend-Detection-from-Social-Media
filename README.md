
# Mental Health Trend Detection from Social Media

An end-to-end Natural Language Processing and Machine Learning project for identifying **potential mental-distress indicators in synthetic social-media posts** and analyzing how these indicators change over time.

> **Important:** This project is intended for educational, research, and portfolio purposes. It does **not** diagnose mental-health conditions, assess individual users, or provide medical advice.

---

## Overview

Mental-health discussions increasingly appear in online communities and social-media environments. This creates opportunities for computational methods to study population-level patterns in language and engagement.

This project explores a reproducible machine-learning workflow that:

1. Loads social-media text data.
2. Cleans and normalizes the text.
3. Converts text into TF-IDF features.
4. Trains a Logistic Regression classifier.
5. Evaluates classification performance.
6. Generates probability-based predictions.
7. Aggregates predictions by date.
8. Calculates daily mental-distress rates.
9. Applies rolling trend analysis.
10. Identifies potential temporal spikes.
11. Produces publication-quality trend visualizations.
12. Saves the trained model and evaluation artifacts.

The repository uses **synthetic data** so that the project can demonstrate the complete analytical workflow without processing real individuals' sensitive social-media content.

---

## Research Question

> **Can NLP-based machine learning identify patterns associated with potential mental-distress indicators in social-media text, and can aggregated predictions reveal changes in these indicators over time?**

---

## Objectives

### Primary objective

Develop a reproducible NLP classification pipeline for distinguishing between:

| Label | Meaning |
|---:|---|
| `0` | Normal / no simulated distress indicator |
| `1` | Potential mental-distress indicator |

### Secondary objectives

- Analyze temporal changes in predicted distress rates.
- Produce daily mental-health trend indicators.
- Calculate smoothed rolling trends.
- Identify potential periods of elevated distress signals.
- Preserve reproducibility through a serialized ML pipeline.
- Separate data processing, modeling, evaluation, and visualization into maintainable modules.

---

## Dataset

The project uses a synthetic social-media dataset.

The repository's dataset is structured around social-media observations containing fields such as:

| Feature | Description |
|---|---|
| `post_id` | Unique post identifier |
| `text` | Simulated social-media text |
| `label` | Binary target variable |
| `date` | Date associated with the post |
| `likes` | Simulated engagement count |
| `shares` | Simulated engagement count |

The dataset is designed to contain realistic textual variation while avoiding the use of identifiable real-world user information.

### Why synthetic data?

Mental-health information is highly sensitive. Using synthetic data allows the project to demonstrate:

- NLP preprocessing
- feature engineering
- machine learning
- evaluation
- temporal analysis
- visualization

without requiring the collection or storage of real people's mental-health-related social-media content.

---

## Machine Learning Architecture

The project uses a classical and interpretable NLP pipeline:

```text
Social Media Text
       │
       ▼
Data Validation
       │
       ▼
Text Cleaning
       │
       ▼
Train/Test Split
       │
       ▼
TF-IDF Vectorization
       │
       ▼
Logistic Regression
       │
       ▼
Predictions + Probabilities
       │
       ├──────────────► Classification Metrics
       │
       ▼
Daily Aggregation
       │
       ▼
Rolling Trend Analysis
       │
       ▼
Mental Health Trend Visualization
```

---

## NLP Processing

The preprocessing pipeline performs:

- Unicode normalization
- lowercasing
- URL removal
- mention removal
- punctuation removal
- whitespace normalization
- missing-text handling
- duplicate-text removal

The cleaned text is stored internally as:

```text
clean_text
```

---

## Feature Engineering

The project uses **Term Frequency-Inverse Document Frequency (TF-IDF)**.

The default configuration includes:

- unigram and bigram features
- English stop-word removal
- maximum feature limit
- minimum document frequency
- maximum document frequency
- sublinear TF scaling

The TF-IDF vectorizer is fitted **only on the training data**.

This prevents information from the test set from leaking into the feature-generation process.

---

## Classification Model

### Logistic Regression

The baseline classifier is Logistic Regression.

Advantages include:

- strong performance for sparse text features
- relatively low computational requirements
- interpretability
- probability estimates
- reproducibility
- easy deployment

The model uses balanced class weighting to reduce sensitivity to class imbalance.

---

## Evaluation

The model is evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Per-class precision
- Per-class recall
- Per-class F1-score
- Confusion matrix

Evaluation results are saved to:

```text
reports/metrics/classification_metrics.json
```

---

## Trend Detection

After generating predictions, the project aggregates them by date.

For each day, the system calculates:

```text
total_posts
distress_posts
distress_rate
distress_rate_percent
```

The distress rate is calculated as:

```text
Distress Rate =
Number of posts predicted as distress
---------------------------------------
Total posts
```

A 7-day rolling average is also calculated to reduce short-term volatility.

---

## Potential Spike Detection

The project includes an analytical mechanism for identifying periods where the observed distress rate is substantially higher than the preceding baseline.

This is intended for **aggregate trend research**, not individual-level risk assessment.

---

## Project Structure

```text
Mental-Health-Trend-Detection-from-Social-Media/
│
├── data/
│   ├── raw/
│   │   └── mental_health_social_media_dataset.csv
│   │
│   └── processed/
│       ├── test_predictions.csv
│       └── mental_health_trends.csv
│
├── models/
│   └── mental_health_classifier.joblib
│
├── reports/
│   ├── figures/
│   │   └── mental_health_trend.png
│   │
│   └── metrics/
│       └── classification_metrics.json
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── model.py
│   ├── evaluation.py
│   ├── trend_analysis.py
│   ├── visualization.py
│   ├── pipeline.py
│   ├── train.py
│   └── predict.py
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/bbjonah/Mental-Health-Trend-Detection-from-Social-Media.git
cd Mental-Health-Trend-Detection-from-Social-Media
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Placement

Place the dataset at:

```text
data/raw/mental_health_social_media_dataset.csv
```

The application expects at least:

```text
text
label
date
```

Additional fields such as:

```text
post_id
likes
shares
```

are supported.

---

## Training

Run:

```bash
python -m src.train
```

The training pipeline will:

1. Load the dataset.
2. Validate the data.
3. Clean the text.
4. Remove duplicate observations.
5. Split the data using stratification.
6. Fit TF-IDF on the training set.
7. Train Logistic Regression.
8. Evaluate the model.
9. Save test predictions.
10. Generate temporal trends.
11. Save the trend dataset.
12. Generate the trend visualization.
13. Serialize the trained ML pipeline.

---

## Generated Artifacts

After training, the following files are generated:

```text
models/
└── mental_health_classifier.joblib
```

```text
reports/
├── figures/
│   └── mental_health_trend.png
│
└── metrics/
    └── classification_metrics.json
```

```text
data/processed/
├── test_predictions.csv
└── mental_health_trends.csv
```

---

## Making a Prediction

Once the model has been trained:

```bash
python -m src.predict \
    --text "I have been feeling overwhelmed and exhausted lately"
```

Example output:

```text
Predicted label: 1
Distress probability: 0.8421
```

The probability represents the model's estimated probability for the project's simulated distress class.

It should **not** be interpreted as a clinical probability or diagnosis.

---

## Reproducibility

The project uses a fixed random seed:

```text
42
```

to make model development and train/test splitting reproducible.

The entire TF-IDF + Logistic Regression workflow is persisted as a single `joblib` artifact.

This means inference uses the same feature transformation learned during training.

---

## Data Leakage Prevention

A key design principle of the project is avoiding train/test contamination.

The correct workflow is:

```text
Raw Data
   │
   ▼
Train/Test Split
   │
   ├── Training Data ──► TF-IDF Fit ──► Model Training
   │
   └── Test Data ─────► TF-IDF Transform ──► Evaluation
```

The TF-IDF vectorizer is therefore never fitted on the test set.

---

## Limitations

This project has important limitations.

### 1. Synthetic data

The dataset does not represent real-world social-media users or actual clinical populations.

### 2. Proxy labels

The binary label represents simulated mental-distress indicators rather than clinical diagnoses.

### 3. Language limitations

The baseline model uses English-language TF-IDF features and may not generalize to multilingual or highly code-switched environments.

### 4. Context limitations

Short social-media posts can lack the context required to understand a person's actual mental state.

### 5. Platform bias

Real social-media datasets may contain substantial demographic, geographic, platform, and sampling biases.

### 6. No clinical interpretation

The model should never be used as a standalone diagnostic or clinical decision-making system.

---

## Ethical Considerations

Mental-health analytics requires strong safeguards.

This project therefore follows several principles:

- No real user mental-health data is required.
- No personally identifiable information is intentionally processed.
- Predictions should be treated as research signals rather than diagnoses.
- Aggregate trend analysis is preferred over individual profiling.
- Model outputs should not be used to make clinical decisions.
- Real-world deployment would require appropriate ethical review, privacy controls, consent considerations, and domain-expert oversight.

---

## Future Improvements

Potential extensions include:

### Advanced NLP

- BERT
- RoBERTa
- DistilBERT
- Sentence Transformers
- contextual embeddings

### Sentiment and emotion analysis

Add:

- sentiment polarity
- emotional intensity
- anxiety-related language
- loneliness indicators
- stress indicators
- sleep-related language
- social-isolation indicators

### Temporal modeling

Potential approaches:

- rolling-window models
- time-series forecasting
- Bayesian change-point detection
- anomaly detection
- temporal transformers

### Explainability

Future versions could include:

- SHAP
- LIME
- TF-IDF feature importance
- model confidence analysis

### Deployment

The project could eventually be deployed as:

- Streamlit dashboard
- FastAPI inference service
- scheduled batch-processing pipeline
- research analytics dashboard

---

## Disclaimer

This project is **not a medical device** and is not intended to diagnose, treat, prevent, or predict a mental-health condition in an individual.

The classifications produced by the model represent computational patterns in synthetic text and should not be interpreted as evidence of a person's actual mental state.

---

## License

This project is released under the MIT License.

See [`LICENSE`](LICENSE) for details.

---

## Author

**Buka Jonah**

GitHub:

```text
https://github.com/bbjonah
```

---

## Project Status

**Status:** Research / Portfolio Project

The current implementation provides a reproducible classical NLP baseline. Future versions can extend the architecture toward transformer-based NLP, richer temporal modeling, explainability, and research-oriented dashboarding.

