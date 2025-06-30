from typing import Dict, Tuple
import copy
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from oceanwave_forecast  import data_manager
from oceanwave_forecast import data_pipeline
from oceanwave_forecast    import models
import mlflow
import random
import matplotlib.pyplot as plt

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




# TODO This plotting func is only for testing, remove it
def plot_random_sample(x_train: torch.Tensor, y_train: torch.Tensor):
    """
    Plot one random example from the train set, overlaying all 3 input features
    and then the target horizon series on the same axis.

    x_train: Tensor of shape (window, batch, features=3)
    y_train: Tensor of shape (horizon, batch)
    """
    window, batch_size, num_feats = x_train.shape
    horizon = y_train.shape[0]
    assert num_feats == 3, "Expected exactly 3 features in x_train"

    # pick a random batch index
    idx = random.randrange(batch_size)

    # extract and convert to numpy
    inp = x_train[:, idx, :].detach().cpu().numpy()  # shape (window, 3)
    tgt = y_train[:, idx].detach().cpu().numpy()     # shape (horizon,)

    # build time axes
    t_in  = np.arange(window)
    t_out = np.arange(window, window + horizon)

    # plot
    fig, ax = plt.subplots()
    ax.plot(t_in, inp[:, 0], label="Feat 0 (input)")
    ax.plot(t_in, inp[:, 1], label="Feat 1 (input)")
    ax.plot(t_in, inp[:, 2], label="Feat 2 (input)")
    ax.plot(t_out, tgt,   label="Target continuation", linewidth=2, linestyle="--")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalized value")
    ax.set_title(f"Sample #{idx}: Inputs + Continuation")
    ax.legend()
    plt.tight_layout()
    plt.show()


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
    model_cfg: Dict
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
        mlflow.log_metric("train_loss", train_loss.item(), step=epoch)
        mlflow.log_metric("val_loss",   val_loss.item(),   step=epoch)

    # 4. Reload best‐seen weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_loss, model