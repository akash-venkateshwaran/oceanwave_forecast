import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, RecurrentNetwork
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, MASE

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor
import torch

from oceanwave_forecast import config, plotting, training

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Data Generation Parameters
RANDOM_SEED = 42
NUM_SAMPLES = 1000
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.85

# Model Architecture Parameters
MAX_ENCODER_LENGTH = 100
MAX_PREDICTION_LENGTH = 20
HIDDEN_SIZE = 64
RNN_LAYERS = 2
CELL_TYPE = "LSTM"
DROPOUT = 0.1

# Training Parameters
BATCH_SIZE = 64
NUM_WORKERS = 0
MAX_EPOCHS = 2500
LEARNING_RATE = 1e-3
GRADIENT_CLIP_VAL = 0.1

# Training Control
LIMIT_TRAIN_BATCHES = 50
VAL_CHECK_INTERVAL = 0.5
LOG_EVERY_N_STEPS = 20

# Early Stopping Parameters
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4

# Target and Feature Definitions
TARGET_VARIABLES = ["T1", "T2"]
COVARIATE_VARIABLES = ["x1", "x2", "x3"]

# Loss Function and Optimizer
LOSS_FUNCTION = SMAPE()
OPTIMIZER = "adamw"

# Callback Parameters
PRINT_METRICS_EVERY_N_EPOCHS = 10

# =============================================================================
# DATA GENERATION
# =============================================================================

def create_multivariate_dataset(seed: int = RANDOM_SEED, num_samples: int = NUM_SAMPLES) -> pd.DataFrame:
    """
    Create synthetic multivariate time series dataset with trends and seasonality.
    
    Args:
        seed: Random seed for reproducibility
        num_samples: Number of time points to generate
        
    Returns:
        DataFrame with time series data
    """
    rng = np.random.default_rng(seed)
    t = np.arange(num_samples)
    
    # Target variables
    T1 = 10 + 2 * t + rng.normal(0, 1, num_samples)
    
    linear_trend = 5 + 0.5 * t
    sine_component = 10 * np.sin(2 * np.pi * t / 50)
    T2 = linear_trend + sine_component + rng.normal(0, 1, num_samples)
    
    # Covariate variables
    x1 = 5 + 0.5 * t + rng.normal(0, 0.2, num_samples)
    x2 = 15 + 0.8 * t + rng.normal(0, 0.3, num_samples)
    
    dec_trend = 50 - 0.2 * t
    sine_exog = 10 * np.sin(2 * np.pi * t / 30)
    x3 = dec_trend + sine_exog + rng.normal(0, 0.4, num_samples)

    df = pd.DataFrame({
        "group_id": 0,           # all rows have same group
        "time_idx": t,
        "T1": T1,
        "T2": T2,
        "x1": x1,
        "x2": x2,
        "x3": x3,
    })
    return df

def prepare_data_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Full dataset
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    cutoff_train = int(df.time_idx.max() * TRAIN_SPLIT)
    cutoff_val = int(df.time_idx.max() * VAL_SPLIT)
    
    train_df = df[df.time_idx <= cutoff_train]
    val_df = df[(df.time_idx > cutoff_train) & (df.time_idx <= cutoff_val)]
    test_df = df[df.time_idx > cutoff_val]
    
    return train_df, val_df, test_df

# =============================================================================
# DATASET CREATION
# =============================================================================

def create_time_series_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Create PyTorch Forecasting TimeSeriesDataSet objects.
    
    Args:
        train_df, val_df, test_df: Data splits
        
    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    train_ds = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET_VARIABLES,
        group_ids=["group_id"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_unknown_reals=TARGET_VARIABLES,
        time_varying_known_reals=COVARIATE_VARIABLES,
        target_normalizer=MultiNormalizer([GroupNormalizer(groups=["group_id"]) for _ in TARGET_VARIABLES]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_df, stop_randomization=True)
    test_ds = TimeSeriesDataSet.from_dataset(train_ds, test_df, stop_randomization=True)
    
    return train_ds, val_ds, test_ds

def create_data_loaders(train_ds, val_ds, test_ds):
    """
    Create PyTorch data loaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = test_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    return train_loader, val_loader, test_loader

# =============================================================================
# MODEL AND TRAINING SETUP
# =============================================================================

def setup_trainer_and_model(train_ds):
    """
    Setup trainer and model with all callbacks and parameters.
    
    Args:
        train_ds: Training dataset for model initialization
        
    Returns:
        Tuple of (trainer, model)
    """
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=EARLY_STOP_MIN_DELTA, 
        patience=EARLY_STOP_PATIENCE, 
        verbose=False, 
        mode="min"
    )
    lr_logger = LearningRateMonitor(logging_interval="epoch")
    print_metrics_callback = training.PrintMetricsCallback(print_every_n_epochs=PRINT_METRICS_EVERY_N_EPOCHS)

    # Trainer
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[lr_logger, early_stop_callback, print_metrics_callback],
        limit_train_batches=LIMIT_TRAIN_BATCHES,
        val_check_interval=VAL_CHECK_INTERVAL,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        enable_progress_bar=True
    )
    
    # Model
    model = RecurrentNetwork.from_dataset(
        train_ds,
        cell_type=CELL_TYPE,
        hidden_size=HIDDEN_SIZE,
        rnn_layers=RNN_LAYERS,
        dropout=DROPOUT,
        optimizer=OPTIMIZER,
        learning_rate=LEARNING_RATE,    
        loss=LOSS_FUNCTION
    )
    
    return trainer, model


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("=== Multivariate Time Series Forecasting ===")
    print(f"Configuration:")
    print(f"  - Targets: {TARGET_VARIABLES}")
    print(f"  - Covariates: {COVARIATE_VARIABLES}")
    print(f"  - Encoder Length: {MAX_ENCODER_LENGTH}")
    print(f"  - Prediction Length: {MAX_PREDICTION_LENGTH}")
    print(f"  - Model: {CELL_TYPE} with {HIDDEN_SIZE} hidden units, {RNN_LAYERS} layers")
    print(f"  - Loss Function: {type(LOSS_FUNCTION).__name__}")
    print()

    # 1. Generate and prepare data
    print("1. Generating synthetic data...")
    df_long = create_multivariate_dataset()
    train_df, val_df, test_df = prepare_data_splits(df_long)
    
    print(f"   - Train samples: {len(train_df)}")
    print(f"   - Validation samples: {len(val_df)}")
    print(f"   - Test samples: {len(test_df)}")

    # 2. Create datasets and data loaders
    print("\n2. Creating datasets and data loaders...")
    train_ds, val_ds, test_ds = create_time_series_datasets(train_df, val_df, test_df)
    train_loader, val_loader, test_loader = create_data_loaders(train_ds, val_ds, test_ds)

    # 3. Plot sample from dataloader (if plotting module is available)
    try:
        plotting.plot_dataloader_sample(train_loader, config.TESTING_REPORTS_DIR / "Test_synthetic_sample.png")
        print("   - Sample plot saved")
    except Exception as e:
        print(f"   - Could not create sample plot: {e}")

    # 4. Setup and train model
    print("\n3. Setting up trainer and model...")
    trainer, model = setup_trainer_and_model(train_ds)
    
    print("\n4. Training model...")
    trainer.fit(model, train_loader, val_loader)

    # 5. Load best model and evaluate
    print("\n5. Evaluating model...")
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = RecurrentNetwork.load_from_checkpoint(best_model_path)
    
    # evaluation_results = training.evaluate_with_model_metrics(best_model, val_loader)
    
    # y_pred = evaluation_results['predictions']
    # y_true = evaluation_results['actuals']
    # model_loss = evaluation_results['loss']

    # print(f"\nModel Performance:")
    # print(f"  - y_pred shape: {y_pred.shape}")
    # print(f"  - y_true shape: {y_true.shape}")
    # print(f"  - Validation Loss: {model_loss:.4f}")

    # 6. Create visualizations
    print("\n6. Creating visualizations...")
    fig = plotting.plot_predictions_with_loss_overlay(best_model, val_loader, trainer)
    plt.show()
    
    print("\nTraining and evaluation completed!")

    return None
    
    # return {
    #     'model': best_model,
    #     'trainer': trainer,
    #     'datasets': (train_ds, val_ds, test_ds),
    #     'data_loaders': (train_loader, val_loader, test_loader),
    #     'metrics': {'validation_loss': model_loss},  # Updated to use the computed loss
    #     'predictions': (y_pred, y_true)
    # }

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    results = main()