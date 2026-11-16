from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
)


def create_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Create the TF-IDF feature extractor.

    The vectorizer is fitted only on training data inside the
    machine-learning pipeline to prevent data leakage.
    """

    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        stop_words="english",
    )
  
