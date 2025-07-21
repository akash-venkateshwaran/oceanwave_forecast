import numpy as np
import pandas as pd

def to_long_df(X_df: pd.DataFrame,
               y_df: pd.DataFrame,
               group: str = "ocean") -> pd.DataFrame:
    """
    Convert a wide‐format timeseries dataset into a “long” format suitable
    for sktime-style multi-series forecasting.

    For each target column in y_df, we:
      1. Copy X_df and add:
         – a constant 'series' column (all set to `group`)
         – a 0-based integer 'time_idx'
      2. Append a 'target' column holding that target’s values
      3. Append a 'tgt_name' column with the target’s original column name

    The result is a single DataFrame with one row per (time_idx × target),
    carrying both features and the corresponding scalar target value.
    """
    # 1️⃣ Copy features and add tracking columns
    base = X_df.copy()
    base["series"]   = group
    base["time_idx"] = np.arange(len(base))

    frames = []
    # 2️⃣ For each target column, attach its values and name
    for tgt in y_df.columns:
        tmp = base.copy()
        tmp["target"]   = y_df[tgt].values.astype(np.float32)
        tmp["tgt_name"] = tgt
        frames.append(tmp)

    # 3️⃣ Concatenate all the per-target frames into one long table
    return pd.concat(frames, ignore_index=True)

# === 1. Create dummy data with 5 time stamps ===
time_index = pd.date_range(start="2025-07-21", periods=5, freq="H")  # 5 hourly timestamps
# Exogenous features: let's say temperature and humidity
X_dummy = pd.DataFrame({
    "temp_C":    [20.0, 21.5, 19.8, 22.1, 20.4],
    "humidity":  [55,  60,   58,   62,   59]
}, index=time_index)

# Two target series: e.g. 'wave_height' and 'wave_period'
y_dummy = pd.DataFrame({
    "wave_height": [1.2, 1.3, 1.1, 1.4, 1.2],
    "wave_period": [5.0, 5.2, 4.9, 5.3, 5.1]
}, index=time_index)

print(X_dummy)

# === 2. Transform both “train” and “test” splits ===
train_df = to_long_df(X_dummy, y_dummy, group="ocean_train")
test_df  = to_long_df(X_dummy, y_dummy, group="ocean_test")

print(train_df)


# === 3. Inspect results ===
print("train_df shape:", train_df.shape)
print("test_df  shape:", test_df.shape)

print("\ntrain_df head:")
print(train_df.head())

print("\ntest_df head:")
print(test_df.head())
