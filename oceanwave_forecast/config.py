from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from sktime.performance_metrics.forecasting import MeanSquaredPercentageError, MeanAbsolutePercentageError

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

# ── Data constants ────────────────────────────────────────────────────────────
TARGETS = ['WVHT', 'APD']

MISSING_MAP = {
            # wind & gust & dominant-period wave:  99 → NaN
            'WSPD': 99.0, 'GST': 99.0, 'DPD': 99.0, 'WVHT': 99.0, 'APD': 99.0,
            # pressure: 9999 → NaN
            'PRES': 9999.0,
            # air, water & dew temps: 999 → NaN
            'ATMP': 999.0, 'WTMP': 999.0, 'DEWP': 999.0,
        }

DROP_COLS = ['TIDE', 'VIS', 'DPD']
CYCLIC_COLS = ['WDIR', 'MWD']

ARIMA_IMPUTE_COLS = ['MWD_sin']

# ── MLflow configuration ─────────────────────────────────────────────────────

MLFLOW_CONFIG_BACKTESTING = {
    "experiment_name": "Oceanwave_Backtesting",
    "tags": {
        "project": "Oceanwave Forecasting",
        "type": "Backtesting",
        "framework": "sktime",
    },
}

SCORERS = [MeanSquaredPercentageError(square_root=True), MeanAbsolutePercentageError()]


# ── Time‐window constants ─────────────────────────────────────────────────────
ONE_DAY = 24
ONE_WEEK = ONE_DAY * 7
HORIZON = ONE_DAY * 3  # The prediction horizon in hours
WINDOW = ONE_WEEK * 3  # The time window for feature engineering in hours

RANDOM_STATE = 42  # For reproducibility

# ── ML Model Configuration ───────────────────────────────────────────────────
XGB_KWARGS = {
        "n_estimators": 800,
        "learning_rate": 0.01,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }


# ── Feature Engg ────────────────────────────────────────────────

EXOG_WINDOWSUMMARY_CONFIG = {
            "lag": [1],
            "mean": [[1, 24], [24, 48]],
        }


TARGET_WINDOWSUMMARY_CONFIG = {               
    "lag":  [1, 2, 3, 4, 24, 48, 72],
    "mean": [[1, 24], [24, 48]],  # rolling‑mean over last day & previous day
}


# ── DeepAR Model Configuration ────────────────────────────────────────────────

DEEPAR_CONFIG = {
    # Model architecture
    "lstm_hidden_dim": 128,
    "lstm_layers": 3,
    "lstm_dropout": 0.2,
    "embedding_dim": 32,
    "num_class": 12,
    
    # Training parameters
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_epochs": 2000,
    "early_stopping_patience": 15,
    "weight_decay": 1e-5,
    
    # Data parameters
    "predict_steps": HORIZON,
    "window_size": WINDOW,
    "predict_start": WINDOW,
    
    # Sampling parameters
    "sample_times": 100,
    
    # Device configuration
    "device": "cuda"
}