from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

START_DATE, END_DATE = '2024-01-01', '2024-06-30'

TARGETS = ['WVHT', 'APD']

MLFLOW_CONFIG_BACKTESTING = {
    "experiment_name": "Oceanwave_Backtesting",
    "tags": {
        "project": "Oceanwave Forecasting",
        "type": "Backtesting",
        "framework": "sktime",
    },
}


# ── Time‐window constants ─────────────────────────────────────────────────────
ONE_DAY         = 24
ONE_WEEK        = ONE_DAY * 7
HORIZON   = ONE_DAY * 3 # The prediction horizon in hours
WINDOW = ONE_WEEK * 3 # The time window for feature engineering in hours

RANDOM_STATE = 42  # For reproducibility
