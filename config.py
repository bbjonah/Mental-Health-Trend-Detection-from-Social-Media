from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"


# ============================================================
# DATA
# ============================================================

DATA_FILE = RAW_DATA_DIR / "mental_health_social_media_dataset.csv"

TEXT_COLUMN = "text"
TARGET_COLUMN = "label"
DATE_COLUMN = "date"

OPTIONAL_COLUMNS = [
    "post_id",
    "likes",
    "shares",
]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True

LOGISTIC_REGRESSION_C = 2.0
LOGISTIC_REGRESSION_MAX_ITER = 2_000


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_FILE = MODELS_DIR / "mental_health_classifier.joblib"
METRICS_FILE = METRICS_DIR / "classification_metrics.json"
TEST_PREDICTIONS_FILE = PROCESSED_DATA_DIR / "test_predictions.csv"
TREND_FILE = PROCESSED_DATA_DIR / "mental_health_trends.csv"
TREND_FIGURE = FIGURES_DIR / "mental_health_trend.png"


# ============================================================
# UTILITIES
# ============================================================

def create_project_directories() -> None:
    """Create required project directories if they do not exist."""

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
      
