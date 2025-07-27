from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sktime.transformations.compose import ColumnEnsembleTransformer
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.forecasting.arima import AutoARIMA
from sklearn.preprocessing import StandardScaler
from sktime.transformations.compose import TransformerPipeline
from sktime.forecasting.trend import PolynomialTrendForecaster
from oceanwave_forecast import config

class MissingValueMarker(BaseEstimator, TransformerMixin):
    """
    Replace specified sentinel values in a DataFrame with NaN."""
    def __init__(self, missing_map: dict):
        """
        missing_map: dict where
            keys   = column names in X,
            values = the sentinel value (or list of values) to replace with NaN.
        """
        if not isinstance(missing_map, dict):
            raise ValueError("missing_map must be a dict: {col_name: sentinel or [sentinels]}")
        self.missing_map = missing_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, sentinels in self.missing_map.items():
            if col not in X.columns:
                print(f"Warning: column '{col}' not in DataFrame; skipping.")
                continue
            # allow a single value or list of values
            X[col] = X[col].replace(sentinels, np.nan)
        return X
    


class DegreeToCyclic(BaseEstimator, TransformerMixin):
    """
    Convert specified columns in a DataFrame from degrees into cyclic (sine and cosine) encoding.
    Out‐of‐range values (<0 or >360) are set to NaN before conversion, and the original columns
    are removed after creating the new features.
    """
    def __init__(self, columns):
        if not isinstance(columns, (list, tuple)):
            raise ValueError("`columns` must be a list or tuple of column names")
        self.columns = list(columns)

    def fit(self, X, y=None):
        # No fitting necessary; stateless transformer
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col not in X.columns:
                print(f"Warning: column '{col}' not found; skipping.")
                continue

            # 1) mask out‐of‐range values
            mask_bad = (X[col] < 0) | (X[col] > 360)
            X.loc[mask_bad, col] = np.nan

            # 2) convert degrees to radians
            radians = np.deg2rad(X[col])

            # 3) create cyclic features
            X[f"{col}_sin"] = np.sin(radians)
            X[f"{col}_cos"] = np.cos(radians)

            # 4) drop the original degree column
            X.drop(columns=col, inplace=True)

        return X


def preprocess_ocean_data(data_ocean):
    """
    Process ocean data using pandas functionality along with existing transformers.
    
    Parameters:
    -----------
    data_ocean : pandas.DataFrame
        Input ocean data with DatetimeIndex
        
    Returns:
    --------
    pandas.DataFrame
        Processed dataframe with all transformations applied
    """
    
    # Remove columns that are not needed for the analysis
    columns_to_drop = config.DROP_COLS
    
    # Imputer the missing values based on the dict map and convert degrees features to cyclic (sin and cos)
    mv_marker = MissingValueMarker(config.MISSING_MAP)
    deg2rad = DegreeToCyclic(columns=config.CYCLIC_COLS)
    
    # Apply transformations step by step
    data_ocean_clean = mv_marker.transform(data_ocean)
    data_ocean_clean = deg2rad.transform(data_ocean_clean)
    
    # Drop specified columns using pandas functionality
    data_ocean_clean = data_ocean_clean.drop(columns=[col for col in columns_to_drop if col in data_ocean_clean.columns])
    
    # Resample to hourly intervals using pandas functionality
    data_ocean_hourly = data_ocean_clean.resample('H').mean()
    
    
    return data_ocean_hourly


def get_pipelines(list_X_cols):
    # 1. Create the Y pipeline 
    pipe_Y = TransformerPipeline(steps=[
    ("imputer",      Imputer(method="ffill")),              # 1) fill NaNs
    # ("deseasonalize", Deseasonalizer(sp=24)),               # 2) de‑seasonalize
    ("scale",        TabularToSeriesAdaptor(StandardScaler())),  # 3) scale
    ])


    # 2. Create the X pipeline

    # Define column groups
    special_cols = config.ARIMA_IMPUTE_COLS
    # season_48_cols = ['MWD_sin', 'MWD_cos']
    # season_24_cols = ['WSPD', 'GST', 'WTMP', 'WDIR_sin']
    # detrend_cols = ['ATMP', 'WTMP', 'DEWP']

    # Identify other columns that are not in special_cols for the ffill imputer
    # Assuming processed_data_X has all the columns, we can get the column names
    all_cols = list_X_cols
    other_cols = [col for col in all_cols if col not in special_cols]

    # Configure AutoARIMA for imputation
    auto_arima = AutoARIMA(
        start_p=1, max_p=3,
        start_q=0, max_q=3,
        d=None,
        seasonal=False,
        stepwise=True,
        information_criterion="aic",
        suppress_warnings=True,
        n_jobs=-1,
    )

    # Step 1: Imputation pipeline
    imputer_pipe = ColumnEnsembleTransformer(
        transformers=[
            # ARIMA imputer only on the special columns
            ("arima_imp", Imputer(method="forecaster", forecaster=auto_arima), special_cols),
            # ffill imputer on all the other columns
            ("ffill_imp", Imputer(method="ffill"), other_cols),
        ],
        remainder="passthrough",  # Pass through any remaining columns
    )

    # Step 2: Deseasonalization pipeline
    # deseason_pipe = ColumnEnsembleTransformer(
    #     transformers=[
    #         # Deseasonalize with period 48 for MWD_sin, MWD_cos
    #         ("deseason_48", Deseasonalizer(sp=48), season_48_cols),
    #         # Deseasonalize with period 24 for WSPD, GST, ATMP, WTMP, WDIR_sin
    #         ("deseason_24", Deseasonalizer(sp=24), season_24_cols),
    #     ],
    #     remainder="passthrough",  # Pass through any remaining columns
    # )

    # # Step 3: Detrend pipeline for ATMP, WTMP, DEWP with polynomial order 4
    # detrend_pipe = ColumnEnsembleTransformer(
    #     transformers=[
    #         ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=4)), detrend_cols),
    #     ],
    #     remainder="passthrough",  # Pass through any remaining columns
    # )

    # Step 4: Standard scaling pipeline for all columns
    scaler_pipe = ColumnEnsembleTransformer(
        transformers=[
            ("scaler", TabularToSeriesAdaptor(StandardScaler()), all_cols),  # Apply to all columns
        ],
        remainder="passthrough",  # This should be empty given we're scaling all columns
    )

    # Combine all steps into the final X pipeline
    pipe_X = TransformerPipeline(
        steps=[
            ("imputer", imputer_pipe),
            # ("deseasonalizer", deseason_pipe),
            # ("detrender", detrend_pipe),
            ("scaler", scaler_pipe),
        ]
    )

    return pipe_X, pipe_Y

