from typing import Dict, Tuple
import copy
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from oceanwave_forecast  import data_manager
from oceanwave_forecast import data_pipeline
from oceanwave_forecast    import models
from oceanwave_forecast.mlflow_utils import MLflowExperimentManager
import random
import matplotlib.pyplot as plt
from sktime.forecasting.base import ForecastingHorizon
from oceanwave_forecast import config, mlflow_utils

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_training_testing_pipeline(
    forecaster,
    model_name,
    run_number,
    y_train_transformed,
    y_test,
    X_feat_train=None,
    X_feat_test=None,
    pipe_Y=None,
    y_differencer=None,                # << NEW
    scorers=None,
    extra_params=None,
    use_exog=False,
):
    """
    Execute a generic training pipeline with MLflow tracking for any forecaster.
    
    Args:
        forecaster: Fitted or unfitted forecaster object
        model_name: Name of the model (e.g., "XGBoost", "Naive")
        run_number: MLflow run number for experiment tracking
        y_train_transformed: Transformed training target DataFrame
        y_test: Test target DataFrame for evaluation
        X_feat_train: Training features DataFrame (if using exogenous variables)
        X_feat_test: Test features DataFrame (if using exogenous variables)
        pipe_Y: Transformer object for inverse-transforming predictions
        scorers: List of scorer callables to compute metrics
        extra_params: Dict of additional parameters to log (optional)
        use_exog: Whether to use exogenous variables (X) for training/prediction
    
    Returns:
        MLflow run object
    """
    # Initialize the experiment manager
    exp_manager = mlflow_utils.MLflowExperimentManager(
        experiment_name=config.MLFLOW_CONFIG_BACKTESTING['experiment_name'],
        run_number=run_number,
        tags=config.MLFLOW_CONFIG_BACKTESTING['tags']
    )

    # Start MLflow run
    run = exp_manager.start_mlflow_run(run_name_prefix=model_name)

    # Check if this is a finished run
    if run.info.status == "FINISHED":
        print(f"🟢 Finished {model_name} run already logged (id={run.info.run_id}) – skipping.")
        return run

    try:
        # Extract and log forecaster parameters
        forecaster_params = forecaster.get_params()
        if forecaster_params:
            exp_manager.log_params(forecaster_params)

        # Log basic parameters
        exp_manager.log_param("model_type", model_name)
        exp_manager.log_param("targets", ",".join(config.TARGETS))

        # Log additional parameters if provided
        if extra_params:
            exp_manager.log_params(extra_params)

        print(f"🔄 Starting {model_name} training...")

        # Fit the forecaster
        if use_exog:
            forecaster.fit(y=y_train_transformed, X=X_feat_train)
            y_pred = forecaster.predict(
                fh=ForecastingHorizon(y_test.index, is_relative=False),
                X=X_feat_test
            )
        else:
            forecaster.fit(y=y_train_transformed)
            y_pred = forecaster.predict(
                fh=ForecastingHorizon(y_test.index, is_relative=False)
            )

        # -------------------------------------------------------------
        #   Inverse transforms:  diff  →  scale
        # -------------------------------------------------------------
        if y_differencer is not None:
            y_pred = y_differencer.inverse_transform(y_pred)
        if pipe_Y is not None:
            y_pred = pipe_Y.inverse_transform(y_pred)


        # Plot and save results for each target
        for target in y_pred.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(y_test[target], label=f"Actual ({target})")
            plt.plot(y_pred[target], label=f"Forecast ({target})", ls="--")
            plt.title(f"{model_name} forecast – {target}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            plot_path = config.REPORTS_DIR / f"Run{run_number}_{model_name}_{target}.png"
            plt.savefig(plot_path)
            exp_manager.log_artifact(str(plot_path), "plots")
            plt.close()


        # Calculate and log metrics
        metrics = {}
        if scorers:
            for scorer in scorers:
                name  = scorer.__class__.__name__
                value = scorer(y_test, y_pred)
                metrics[name] = value
                print(f"{name}: {value:.4f}")
            exp_manager.log_metrics(metrics)

        print(f"✅  {model_name} run complete.")
    finally:
        exp_manager.end_mlflow_run()

    return run





def make_datasets(
    data_cfg: Dict,
    model_cfg: Dict
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    """
    Builds (x_train, y_train), (x_val, y_val) by reading from data_cfg and slicing
    windows/horizons according to model_cfg.

    data_cfg must contain:
        - "countries": List[str]
        - any other keys your data_pipeline uses (e.g. "csv_path", "date_column", "target_column")

    model_cfg must contain:
        - "window": int
        - "horizon": int
        - "val_split": float
    """

    # 1. Load the raw DataFrame (CSV path etc. should be encoded in data_manager)
    df = data_manager.covid_until_feb_2021()
    total_rows       = df.shape[0]
    unique_countries = df["Country"].nunique()
    print(f"Loaded DataFrame → rows = {total_rows}, unique countries = {unique_countries}")

    X_all, Y_all = [], []

    # 2. For each country, compute diffs, features, then sliding windows
    for c in data_cfg["countries"]:
        ts = df[df["Country"] == c]["Confirmed"].diff().dropna().values
        print(f"[{c}] raw diffs → ts.shape = {ts.shape}")

        feats = data_pipeline.make_feature_matrix(ts)
        print(f"[{c}] feature matrix → feats.shape = {feats.shape}")

        X_c, Y_c = data_pipeline.sliding_window(
            feats,
            model_cfg["window"],
            model_cfg["horizon"]
        )
        if X_c.size:
            print(
                f"[{c}] sliding_window → "
                f"X_c.shape = {X_c.shape}; "
                f"Y_c.shape = {Y_c.shape}"
            )
        else:
            print(f"[{c}] sliding_window → no samples")

        X_all.append(X_c)
        Y_all.append(Y_c)

    # 3. Concatenate all countries along the first dimension
    X_cat = np.concatenate(X_all, axis=0)  # → shape: (sum_of_samples, window, num_features)
    Y_cat = np.concatenate(Y_all, axis=0)  # → shape: (sum_of_samples, horizon, 1)

    # 4. Convert to torch Tensors and transpose to (window, batch, features) and (horizon, batch)
    x = torch.from_numpy(X_cat).float().transpose(0, 1)
    y = torch.from_numpy(Y_cat).float().transpose(0, 1)[:, :, 0]

    print(f"After stacking all countries → x.shape = {x.shape}, y.shape = {y.shape}")

    # 5. Random train/val split along the batch dimension
    batch_size = x.shape[1]
    mask_val   = torch.rand(batch_size) < model_cfg["val_split"]
    n_val      = int(mask_val.sum().item())
    n_train    = batch_size - n_val
    print(f"Splitting           → train examples = {n_train}, val examples = {n_val}")

    x_train = x[:, ~mask_val]
    y_train = y[:, ~mask_val]
    x_val   = x[:,  mask_val]
    y_val   = y[:,  mask_val]

    print(
        f"Train set           → x_train.shape = {x_train.shape}, "
        f"y_train.shape = {y_train.shape}"
    )
    print(
        f"Val   set           → x_val.shape   = {x_val.shape},   "
        f"y_val.shape   = {y_val.shape}"
    )

    return (x_train, y_train), (x_val, y_val)



def train_once(
    data_cfg: Dict,
    model_cfg: Dict,
    mlflow_manager: MLflowExperimentManager
) -> Tuple[float, nn.Module]:
    """
    Trains a Seq2Seq model using:
      - Data parameters from data_cfg
      - Model/hyperparameters from model_cfg

    Returns:
       (best_val_loss, best_model)
    """

    # 1. Build datasets (x_tr, y_tr), (x_val, y_val)
    (x_tr, y_tr), (x_val, y_val) = make_datasets(data_cfg, model_cfg)
    x_tr, y_tr, x_val, y_val = [
        t.to(_DEVICE) for t in (x_tr, y_tr, x_val, y_val)
    ]

    # 2. Instantiate Seq2Seq (assumes your models.Seq2Seq constructor unchanged)
    model = models.Seq2Seq(
        hidden_size = model_cfg["hidden_size"],
        hidden_fc   = model_cfg["hidden_fc"],
        input_size  = 3,
        output_size = 1
    ).to(_DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=model_cfg["lr"])
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None

    # 3. Training‐Validation loop
    for epoch in range(1, model_cfg["epochs"] + 1):
        # ---- (a) Training step ----
        model.train()
        optimizer.zero_grad()
        out = model(
            x_tr,
            model_cfg["horizon"],
            model_cfg["teacher_ratio"],
            y_tr
        )
        train_loss = criterion(out.squeeze(2), y_tr)
        train_loss.backward()
        optimizer.step()

        # ---- (b) Validation step ----
        model.eval()
        with torch.no_grad():
            val_pred = model.predict(x_val, model_cfg["horizon"])
            val_loss = criterion(val_pred.squeeze(2), y_val)

        # ---- (c) Track best model state ----
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state    = copy.deepcopy(model.state_dict())

        # ---- (d) Log metrics to MLflow (if you are using it) ----
        mlflow_manager.log_metric("train_loss", train_loss.item(), step=epoch)
        mlflow_manager.log_metric("val_loss",   val_loss.item(),   step=epoch)

    # 4. Reload best‐seen weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_loss, model