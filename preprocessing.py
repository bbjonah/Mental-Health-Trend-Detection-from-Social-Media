import re
import unicodedata

import pandas as pd

from src.config import DATE_COLUMN, TEXT_COLUMN


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Normalize social-media text.

    The function intentionally keeps alphabetic content while removing
    URLs, mentions and excessive whitespace.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalization.
    text = unicodedata.normalize("NFKC", text)

    # Lowercase.
    text = text.lower()

    # Remove URLs.
    text = URL_PATTERN.sub(" ", text)

    # Remove user mentions.
    text = MENTION_PATTERN.sub(" ", text)

    # Remove punctuation/symbols while preserving whitespace.
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Collapse repeated whitespace.
    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to a copy of the dataframe.
    """

    processed = df.copy()

    processed[TEXT_COLUMN] = processed[TEXT_COLUMN].fillna("").astype(str)

    processed["clean_text"] = processed[TEXT_COLUMN].apply(clean_text)

    processed[DATE_COLUMN] = pd.to_datetime(
        processed[DATE_COLUMN],
        errors="coerce",
    )

    processed = processed.dropna(
        subset=["clean_text", DATE_COLUMN]
    )

    processed = processed[
        processed["clean_text"].str.len() > 0
    ].copy()

    return processed


def remove_duplicate_posts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate text observations.

    This is useful for preventing duplicated synthetic or repeated
    observations from disproportionately influencing the model.
    """

    return df.drop_duplicates(
        subset=["clean_text"],
        keep="first",
    ).reset_index(drop=True)
  
