

import os
import joblib
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.tseries.frequencies import to_offset
from sktime.forecasting.base import ForecastingHorizon

        
def summarize_cv_score(scores, results):
    """
    Summarize cross-validation scores from the evalute() for each fold and each target variable. """
    # Initialize accumulator dictionary:
    # {scorer_name: {column: [score values per fold, ...]}}
    accum = {}
    # Assume the first fold's y_test has the list of columns
    cols = results.iloc[0]["y_test"].columns
    for scorer in scores:
        scorer_name = scorer.__class__.__name__
        accum[scorer_name] = {col: [] for col in cols}
    
    # Loop over each fold in results
    for i, row in results.iterrows():
        yt = row["y_test"]
        yp = row["y_pred"]
        print(f"Fold {i}:")
        for scorer in scores:
            scorer_name = scorer.__class__.__name__
            for col in cols:
                score_val = scorer(yt[col], yp[col])
                print(f"  {scorer_name} for {col}: {score_val:.4f}")
                accum[scorer_name][col].append(score_val)
    
    # Print mean scores over all folds
    print("Mean scores over all folds:")
    for scorer_name, d in accum.items():
        for col, scores_list in d.items():
            mean_score = np.mean(scores_list)
            print(f"  {scorer_name} for {col}: {mean_score:.4f}")

def get_best_fold(results, metric='test_MeanSquaredPercentageError'):
    """
    Identify the best fold index, error, and fitted model from CV results.
    
    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame containing cross-validation results, including:
          - one column named exactly `metric`
          - column 'fitted_forecaster'
          - columns 'y_train', 'y_test', 'y_pred'
    metric : str
        The column name in `results` to use for selecting the best fold.
    
    Returns
    -------
    best_fold_idx : int
    best_model : object
    best_fold_error : float
    y_train, y_test, y_pred : pandas.Series or DataFrame
    """
    # pick metric column
    if metric not in results.columns:
        print(f"Warning: Metric '{metric}' not found; defaulting to fold 0")
        best_fold_idx = 0
        best_fold_error = None
    else:
        vals = results[metric]
        # handle multi-output metrics
        if isinstance(vals.iloc[0], (list, np.ndarray)):
            mean_errs = [np.mean(v) for v in vals]
            best_fold_idx = int(np.argmin(mean_errs))
            best_fold_error = mean_errs[best_fold_idx]
        else:
            best_fold_idx = int(vals.argmin())
            best_fold_error = float(vals.min())
        print(f"Best fold: {best_fold_idx} with {metric} = {best_fold_error:.4f}")
    
    # get the fitted forecaster
    if 'fitted_forecaster' not in results.columns:
        raise KeyError("'fitted_forecaster' column is missing in results")
    best_model = results['fitted_forecaster'].iloc[best_fold_idx]
    
    # grab data for plotting later
    y_train = results.get('y_train', pd.Series()).iloc[best_fold_idx]
    y_test  = results.get('y_test', pd.Series()).iloc[best_fold_idx]
    y_pred  = results.get('y_pred', pd.Series()).iloc[best_fold_idx]
    
    return best_fold_idx, best_model, best_fold_error, y_train, y_test, y_pred


def export_bestmodel_and_results(results, output_dir,
                                 metric='test_MeanSquaredPercentageError'):
    """
    Uses get_best_fold() to pick the best fold and then exports model, CSV and plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find best fold
    idx, model, error, y_train, y_test, y_pred = get_best_fold(results, metric)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mtype = type(model).__name__
    
    # Save model
    model_path = os.path.join(output_dir, f"{ts}_{mtype}_model.pkl")
    joblib.dump(model, model_path)
    
    # Save full CV results
    csv_path = os.path.join(output_dir, f"{ts}_{mtype}_cv_results.csv")
    results.to_csv(csv_path, index=False)
    
    print(f"Model saved to: {model_path}")
    print(f"CV results saved to: {csv_path}")
    
    # Plot each target
    if hasattr(y_pred, 'columns'):
        targets = y_pred.columns
    else:
        targets = [None]
    
    for tgt in targets:
        plt.figure(figsize=(12, 6))
        if y_train is not None:
            plt.plot(y_train[tgt], label=f"Train {tgt}")
        plt.plot(y_test[tgt], label=f"Actual {tgt}")
        plt.plot(y_pred[tgt], label=f"Pred {tgt}", linestyle='--')
        plt.title(f"{mtype} Forecast (fold {idx})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir,
                                 f"{ts}_{mtype}_{tgt or 'series'}_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to: {plot_path}")
    
    return idx, model




def forecast_with_exog(best_model, X_train_transformed: pd.DataFrame, X_test_transformed: pd.DataFrame):
    """
    Generate out‑of‑sample forecasts aligned with X_test_transformed.index. and output the test period only.

    Parameters
    ----------
    best_model : sktime forecaster
        A fitted forecaster with attribute `cutoff` (e.g. from a CV run).
    X_train_transformed : pd.DataFrame
        Exogenous features used to fit the model.
    X_test_transformed : pd.DataFrame
        Exogenous features for which you want predictions.

    Returns
    -------
    y_pred : pd.Series or pd.DataFrame
        Forecasts indexed the same as X_test_transformed.
    """
    # helper to infer a pandas-compatible frequency string
    def infer_freq(idx):
        return idx.freq or pd.infer_freq(idx) or "H"

    # 1. build full out‑of‑sample index
    cutoff = best_model.cutoff[0]                       # Timestamp of last training point
    print(f"Models Cutoff Date: {cutoff}")
    print(f"X_test Start Date: {X_test_transformed.index[0]}")
    freq   = infer_freq(X_test_transformed.index)       # e.g. 'H', 'D', ...
    start  = cutoff + to_offset(freq)                   # one step after cutoff
    end    = X_test_transformed.index[-1]
    full_idx = pd.date_range(start=start, end=end, freq=freq)

    # 2. stitch together train+test exogenous and reindex to full_idx
    X_full = (
        pd.concat([X_train_transformed, X_test_transformed])
          .sort_index()
          .reindex(full_idx)
    )
    if len(X_full) != len(full_idx):
        raise ValueError(f"X_full length {len(X_full)} ≠ full_idx length {len(full_idx)}")

    # 3. build a relative ForecastingHorizon covering the full OOS span
    fh_rel = ForecastingHorizon(np.arange(1, len(full_idx) + 1), is_relative=True)

    # 4. predict once for the whole span
    y_pred_full = best_model.predict(fh=fh_rel, X=X_full)

    # 5. slice back to just the test timestamps (Outputs only the test period)
    y_pred = y_pred_full.reindex(X_test_transformed.index)

    if not y_pred.index.equals(X_test_transformed.index):
        raise AssertionError("Output index mismatch!")

    return y_pred



import xgboost as xgb

# exponential decay: every 10 boosting rounds multiply LR by 0.95
def lr_schedule(current_iter: int) -> float:
    base_lr = 0.1
    return base_lr * np.power(0.97, current_iter)