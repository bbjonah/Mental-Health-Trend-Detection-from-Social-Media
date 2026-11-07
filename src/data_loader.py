from pathlib import Path

import pandas as pd

from src.config import (
    DATA_FILE,
    DATE_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
)


REQUIRED_COLUMNS = {
    TEXT_COLUMN,
    TARGET_COLUMN,
    DATE_COLUMN,
}


def load_dataset(path: Path = DATA_FILE) -> pd.DataFrame:
    """
    Load the social-media dataset.

    Parameters
    ----------
    path:
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.
    ValueError
        If required columns are missing.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Place the CSV file inside data/raw/."
        )

    df = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate basic dataset assumptions.
    """

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[TEXT_COLUMN].isna().all():
        raise ValueError("Text column contains no usable observations.")

    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            "Target variable must contain at least two classes."
        )


def load_and_validate(path: Path = DATA_FILE) -> pd.DataFrame:
    """Load and validate the dataset."""

    df = load_dataset(path)
    validate_dataset(df)

    return df
  
