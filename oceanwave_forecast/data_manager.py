import pandas as pd
from typing import Tuple, List
import matplotlib.pyplot as plt
from calendar import monthrange
from sktime.param_est.stationarity import StationarityADF

from sktime.param_est.stationarity import StationarityADF
from collections import namedtuple
import warnings
import pymannkendall as mk
import numpy as np
import math
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelmax
from scipy.stats import norm

def extract_raw_data(txt_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Read a NOAA-style space-delimited .txt file with a header line beginning '#',
    parse into a DataFrame with a datetime index, print shape & summary, and
    optionally export to CSV.

    Parameters
    ----------
    txt_path : str
        Path to the raw .txt file.
    output_path : str, optional
        If provided, path to write the DataFrame as CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime, containing all measurement columns.
    """
    # 1) Read and parse header line
    with open(txt_path, 'r') as f:
        header_line = f.readline().lstrip('#').strip()
    col_names = header_line.split()

    # 2) Load the rest of the data
    df = pd.read_csv(
        txt_path,
        delim_whitespace=True,
        comment='#',
        header=None,
        names=col_names,
        dtype={col: float for col in col_names[5:]}  # numeric for freqs
    )

    # 3) Build a single datetime column and set as index
    df['datetime'] = pd.to_datetime({
        'year':   df[col_names[0]].astype(int),
        'month':  df[col_names[1]].astype(int),
        'day':    df[col_names[2]].astype(int),
        'hour':   df[col_names[3]].astype(int),
        'minute': df[col_names[4]].astype(int),
    })
    df = df.set_index('datetime')
    df = df.drop(columns=col_names[:5])

    print("DataFrame shape:", df.shape)
    print("\nInfo:")
    df.info()
    print("\nDescriptive statistics:")
    print(df.describe())

    if output_path:
        df.to_csv(output_path)
        print(f"\nExported DataFrame to: {output_path}")

    return df


def DATAF_separate_columns(df: pd.DataFrame, output_col: str = None) -> Tuple[List[str], List[str]]:
    """Separate DataFrame columns into numerical and categorical lists."""
    cols = df.columns.tolist()
    input_cols = [col for col in cols if col != output_col]
    numerical_cols = df[input_cols].select_dtypes(include=['number']).columns.tolist()
    categorical_cols = [col for col in input_cols if col not in numerical_cols]
    return numerical_cols, categorical_cols

def DATAF_separate_num_columns(df: pd.DataFrame, num_cols: List[str], unique_threshold: int = 20) -> Tuple[List[str], List[str]]:
    """
    Separates a list of numerical columns into discrete and continuous lists based on the number of unique values.
    """
    discrete_cols = []
    continuous_cols = []
    
    for col in num_cols:
        # Count unique values. By default nunique() ignores NaNs.
        unique_count = df[col].nunique()
        if unique_count < unique_threshold:
            discrete_cols.append(col)
        else:
            continuous_cols.append(col)
    
    return discrete_cols, continuous_cols



def plot_monthly_seasonal_components(
    seasonal: pd.DataFrame,
    year: int,
    months: list[int],
    color_map: dict[str,str] | None = None,
    figsize: tuple[int,int] = (14, 10)
):
    """
    Plot each month’s seasonal components on its own row, 
    with an x–axis that runs from the 1st of the month onward.
    
    seasonal : DataFrame whose index is a DateTimeIndex and whose
               columns are the various seasonal components.
    year     : int, the calendar year to select.
    months   : list of ints, the months to plot (1 = Jan, …, 12 = Dec).
    color_map: optional dict mapping component names → colors.
    """
    # infer a uniform frequency for all sub‐series
    freq = seasonal.index.freq or pd.infer_freq(seasonal.index)
    n_months = len(months)
    
    # use a default palette if none supplied
    if color_map is None:
        color_map = {comp: None for comp in seasonal.columns}
    
    fig, axes = plt.subplots(nrows=n_months, figsize=figsize, sharex=True)
    # if only one subplot, wrap axes in a list for uniform indexing
    if n_months == 1:
        axes = [axes]
    
    for ax, m in zip(axes, months):
        # slice out the month
        dfm = seasonal[
            (seasonal.index.year  == year) &
            (seasonal.index.month == m)
        ]
        # build a dummy datetime index from the 1st, same length & freq
        temp_idx = pd.date_range(
            start=pd.Timestamp(f"{year}-01-01"),
            periods=len(dfm),
            freq=freq
        )
        # plot each component
        for comp in seasonal.columns:
            ax.plot(
                temp_idx,
                dfm[comp].values,
                label=comp,
                color=color_map.get(comp),
            )
        ax.set_title(f"Seasonal Components — {year}-{m:02d}")
        ax.set_ylabel("Seasonal Effect")
        ax.legend(ncol=2, fontsize="small")
    
    # only label the bottom x‐axis
    axes[-1].set_xlabel("Day of Month")
    plt.tight_layout()
    plt.show()


def plot_daily_seasonal_components(
    seasonal: pd.DataFrame,
    year: int,
    month: int,
    days: list[int],
    color_map: dict[str, str] | None = None,
    figsize: tuple[int, int] = (14, 5)
):
    """
    Plot each specified day's seasonal components on its own subplot row,
    with an x-axis that runs from hour 0 to 23.

    seasonal : DataFrame whose index is a DateTimeIndex and whose
               columns are the various seasonal components.
    year     : int, the calendar year to select.
    month    : int, the month to select (1 = Jan, …, 12 = Dec).
    days     : list of ints, the days of the month to plot.
    color_map: optional dict mapping component names → colors.
    figsize  : size of the overall figure (width, height).
    """
    # Validate days
    _, max_day = monthrange(year, month)
    invalid = [d for d in days if d < 1 or d > max_day]
    if invalid:
        print(f"Error: Day(s) {invalid} out of range for {year}-{month:02d} (1 to {max_day}).")
        return

    # Use default colors if none supplied
    if color_map is None:
        color_map = {comp: None for comp in seasonal.columns}

    n_days = len(days)
    fig, axes = plt.subplots(nrows=n_days, figsize=figsize, sharex=True)
    if n_days == 1:
        axes = [axes]

    for ax, day in zip(axes, days):
        # slice the day
        df_day = seasonal[
            (seasonal.index.year  == year) &
            (seasonal.index.month == month) &
            (seasonal.index.day   == day)
        ]
        if df_day.empty:
            ax.text(0.5, 0.5, f"No data for {year}-{month:02d}-{day:02d}", ha='center')
            ax.set_title(f"{year}-{month:02d}-{day:02d}")
            continue

        hours = df_day.index.hour
        for comp in seasonal.columns:
            ax.plot(
                hours,
                df_day[comp].values,
                label=comp,
                color=color_map.get(comp),
            )
        ax.set_title(f"Seasonal Components — {year}-{month:02d}-{day:02d}")
        ax.set_ylabel("Seasonal Effect")
        ax.set_xticks(range(0, 24, 3))
        ax.legend(ncol=2, fontsize="small")

    axes[-1].set_xlabel("Hour of Day")
    plt.tight_layout()
    plt.show()





def test_stationarity(time_series, p_threshold=0.05, maxlag=50, regression='c', autolag='AIC', plot=False):
    """
    Test the stationarity of a time series using Augmented Dickey-Fuller test.
    
    Parameters:
    -----------
    time_series : pandas.Series
        The time series to test for stationarity
    p_threshold : float, optional (default=0.05)
        Significance threshold for the test
    maxlag : int or None, optional (default=None)
        Maximum lag to be used in the test
    regression : str, optional (default='c')
        Type of regression to include:
        'c': constant only
        'ct': constant and trend
        'ctt': constant, linear and quadratic trend
        'n': no constant, no trend
    autolag : str or None, optional (default='AIC')
        Method to automatically determine lag length
    plot : bool, optional (default=True)
        Whether to plot the time series and its rolling statistics
        
    Returns:
    --------
    dict : Dictionary containing test results
    """
    # Create and fit the stationarity test
    stationarity_test = StationarityADF(
        p_threshold=p_threshold, 
        maxlag=maxlag, 
        regression=regression, 
        autolag=autolag
    )
    stationarity_test.fit(time_series)
    
    # Get test results
    results = stationarity_test.get_fitted_params()
    
    # Print comprehensive results
    print("\n=== Augmented Dickey-Fuller Test Results ===")
    print(f"ADF Test Statistic: {results['test_statistic']:.4f}")
    print(f"Results: {results}")
    
    conclusion = "Stationary" if results['stationary'] else "Non-stationary"
    print(f"\nConclusion: The time series is {conclusion} at {p_threshold*100}% significance level.")
    print(f"Null Hypothesis {'rejected' if results['stationary'] else 'failed to reject'}")
    
    # Visualize the time series and rolling statistics if requested
    if plot:
        plt.figure(figsize=(12, 6))
        
        # Original Series
        plt.subplot(211)
        plt.plot(time_series, label='Original Series')
        plt.title('Time Series')
        plt.legend()
        
        # Rolling mean and std
        rolling_mean = time_series.rolling(window=12).mean()
        rolling_std = time_series.rolling(window=12).std()
        
        plt.subplot(212)
        plt.plot(rolling_mean, label='Rolling Mean (window=12)')
        plt.plot(rolling_std, label='Rolling Standard Deviation (window=12)')
        plt.title('Rolling Statistics')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    return results




def plot_before_after_imputation(original_series, imputed_series, column_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the original data with missing values highlighted
    ax.plot(original_series.index, original_series, 'b-', label='Original Data', alpha=0.7)
    missing_mask = original_series.isna()
    
    # Plot imputed values in red
    if sum(missing_mask) > 0:
        imputed_points = pd.Series(index=original_series.index[missing_mask], 
                                 data=imputed_series[missing_mask])
        ax.plot(imputed_points.index, imputed_points, 'ro', label='Imputed Values')
    
    ax.set_title(f'Imputation Results for {column_name}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.legend()
    plt.tight_layout()
    return fig


# Function to check for trend using Mann-Kendall test:

def check_trend(y, confidence=0.05, seasonal_period=None,prewhiten=None):
    
    name, slope, p, trend, direction = _check_mann_kendall(y, confidence, seasonal_period, prewhiten)
    
    res = namedtuple(name, ["trend", "direction", "slope", "p_value"])
    return res(trend, direction, slope, p)

def _check_mann_kendall(y, confidence=0.05, seasonal_period=None, prewhiten=None):
    
    if prewhiten is None:
        if len(y)<50:
            prewhiten = True
        else:
            prewhiten = False
    else:
        if not prewhiten and len(y)<50:
            warnings.warn("For timeseries with < 50 samples, it is recommended to prewhiten the timeseries. Consider passing `prewhiten=True`")
        if prewhiten and len(y)>50:
            warnings.warn("For timeseries with > 50 samples, it is not recommended to prewhiten the timeseries. Consider passing `prewhiten=False`")
    y = _check_convert_y(y)
    if seasonal_period is None:
        if prewhiten:
            _res = mk.pre_whitening_modification_test(y, alpha=confidence)
        else:
            _res = mk.original_test(y, alpha=confidence)
    else:
        _res = mk.seasonal_test(y, alpha=confidence, period=seasonal_period)
    trend=True if _res.p<confidence else False
    if _res.slope>0:
        direction="increasing"
    else:
        direction="decreasing"
    return type(_res).__name__,_res.slope, _res.p, trend, direction

def _check_convert_y(y):
    assert not np.any(np.isnan(y)), "`y` should not have any nan values"
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values.squeeze()
    assert y.ndim==1
    return y


def check_heteroscedastisticity(y, confidence=0.05):
    y = _check_convert_y(y)
    res = namedtuple("White_Test", ["heteroscedastic", "lm_statistic", "lm_p_value"])
    #Fitting a linear trend regression
    x = np.arange(len(y))
    x = sm.add_constant(x)
    model = sm.OLS(y,x)
    results = model.fit()
    lm_stat, lm_p_value, f_stat, f_p_value = het_white(results.resid, x)
    if lm_p_value<confidence and f_p_value < confidence:
        hetero = True
    else:
        hetero = False
    return res(hetero, lm_stat, lm_p_value)



def check_seasonality(y, max_lag=24, seasonal_period=None, confidence=0.05, verbose=True):

    '''To determine if the seasonality is statistically significant, the function:

            Computes autocorrelation values for different lags
            Finds local maxima in these values (potential seasonal periods)
            Uses Bartlett's formula to calculate confidence intervals
            Tests if the ACF at the candidate seasonal period exceeds the significance threshold

    If a user specifies a particular period to check (via seasonal_period), the function only tests that specific period. Otherwise, it checks all detected local maxima in the ACF.'''

    res = namedtuple("Seasonality_Test", ["seasonal", "seasonal_periods"])
    y = _check_convert_y(y)
    if seasonal_period is not None and (seasonal_period < 2 or not isinstance(seasonal_period, int)):
        raise ValueError('seasonal_period must be an integer greater than 1.')

    if seasonal_period is not None and seasonal_period >= max_lag:
        raise ValueError('max_lag must be greater than seasonal_period.')

    n_unique = np.unique(y).shape[0]

    if n_unique == 1:  # Check for non-constant TimeSeries
        return res(False, 0)
    r = acf(y, nlags=max_lag, fft=False)  # In case user wants to check for seasonality higher than 24 steps.

    # Finds local maxima of Auto-Correlation Function
    candidates = argrelmax(r)[0]
    print(candidates)

    if len(candidates) == 0:
        if verbose:
            print('The ACF has no local maximum for m < max_lag = {}. Try larger max_lag'.format(max_lag))
        return res(False, 0)

    if seasonal_period is not None:
        # Check for local maximum when m is user defined.
        test = seasonal_period not in candidates

        if test:
            return res(False, seasonal_period)

        candidates = [seasonal_period]

    # Remove r[0], the auto-correlation at lag order 0, that introduces bias.
    r = r[1:]

    # The non-adjusted upper limit of the significance interval.
    band_upper = r.mean() + norm.ppf(1 - confidence / 2) * r.var()

    # Significance test, stops at first admissible value. The two '-1' below
    # compensate for the index change due to the restriction of the original r to r[1:].
    for candidate in candidates:
        stat = _bartlett_formula(r, candidate - 1, len(y))
        if r[candidate - 1] > stat * band_upper:
            return res(True, candidate)
    return res(False, 0)

def _bartlett_formula(r: np.ndarray,
                      m: int,
                      length: int) -> float:
    """
    Computes the standard error of `r` at order `m` with respect to `length` according to Bartlett's formula.
    Parameters
    ----------
    r
        The array whose standard error is to be computed.
    m
        The order of the standard error.
    length
        The size of the underlying sample to be used.
    Returns
    -------
    float
        The standard error of `r` with order `m`.
    """

    if m == 1:
        return math.sqrt(1 / length)
    else:
        return math.sqrt((1 + 2 * sum(map(lambda x: x ** 2, r[:m - 1]))) / length)
