
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
import matplotlib.pyplot as plt
from sktime.forecasting.base import ForecastingHorizon
from oceanwave_forecast import config, mlflow_utils, plotting
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sktime.forecasting.base import ForecastingHorizon
from typing import List, Dict, Optional, Tuple, Any


import torch
import numpy as np
from typing import Dict, Tuple, Optional

from lightning.pytorch.callbacks import Callback



def create_data_blocks(y_test: pd.DataFrame, horizon: int) -> List[pd.DataFrame]:
    """Split test data into blocks of size horizon."""
    blocks = []
    n_samples = len(y_test)
    
    for i in range(0, n_samples, horizon):
        block = y_test.iloc[i:i + horizon]
        if len(block) == horizon:  # Only include complete blocks
            blocks.append(block)
    
    return blocks

def predict_block(forecaster, y_train_final: pd.DataFrame, block: pd.DataFrame, 
                 X_test_final: Optional[pd.DataFrame] = None, use_exog: bool = False) -> pd.DataFrame:
    """Make predictions for a single block."""
    fh = ForecastingHorizon(block.index, is_relative=False)
    
    if use_exog and X_test_final is not None:
        X_block = X_test_final.loc[block.index]
        return forecaster.predict(fh=fh, X=X_block)
    else:
        return forecaster.predict(fh=fh)

def compute_block_scores(y_true_block: pd.DataFrame, y_pred_block: pd.DataFrame, 
                        scorers: List[Any]) -> Dict[str, float]:
    """Compute scores for a single block."""
    scores = {}
    for scorer in scorers:
        name = scorer.__class__.__name__
        scores[name] = scorer(y_true_block, y_pred_block)
    return scores

def aggregate_scores(block_scores: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate scores across blocks by taking the mean."""
    if not block_scores:
        return {}
    
    aggregated = {}
    score_names = block_scores[0].keys()
    
    for name in score_names:
        values = [scores[name] for scores in block_scores]
        aggregated[f"avg_{name}"] = np.mean(values)
        aggregated[f"std_{name}"] = np.std(values)
    
    return aggregated


def process_blocks_iteratively(forecaster, y_train_final: pd.DataFrame, 
                              y_test_blocks: List[pd.DataFrame],
                              X_test_final: Optional[pd.DataFrame] = None,
                              use_exog: bool = False) -> List[pd.DataFrame]:
    """Process blocks iteratively, updating training data with previous predictions."""
    y_pred_blocks = []
    current_train = y_train_final.copy()
    
    for block in y_test_blocks:
        # Predict current block
        y_pred_block = predict_block(forecaster, current_train, block, X_test_final, use_exog)
        y_pred_blocks.append(y_pred_block)
        
        # Update training data with actual values for next iteration
        # In practice, you might want to use predictions instead for true recursive forecasting
        current_train = pd.concat([current_train, block])
    
    return y_pred_blocks

def run_training_testing_pipeline(
    forecaster,
    model_name: str, 
    run_number: int,
    y_train: pd.DataFrame,           
    y_test: pd.DataFrame,            
    X_train: Optional[pd.DataFrame] = None,      
    X_test: Optional[pd.DataFrame] = None,       
    pipe_Y = None,       
    pipe_X = None,       
    feat_pipeline = None, 
    differencer = None,  
    scorers = None,
    extra_params = None,
    use_exog: bool = False,
    horizon: int = None,  # Add horizon parameter
    recursive_blocks: bool = False  # Whether to use recursive block processing
):
    """
    
    Complete training pipeline with block-wise processing for long time series.
    
    BLOCK-WISE PROCESSING:
    When y_test length > horizon, the test data is divided into consecutive blocks of size 'horizon'.
    Each block represents a separate forecasting window where the model predicts 'horizon' steps ahead.
    
    SCORING METHODOLOGY:
    - Individual scores are computed for each block independently
    - Aggregated scores are calculated as:
      * avg_[metric]: Mean of the metric across all blocks
      * std_[metric]: Standard deviation of the metric across all blocks
    
    This approach evaluates model consistency across multiple forecasting horizons and provides
    robust performance metrics that account for temporal variations in forecast accuracy.
    """
    
    # Initialize MLflow (assuming your existing setup)
    exp_manager = mlflow_utils.MLflowExperimentManager(
        experiment_name=config.MLFLOW_CONFIG_BACKTESTING['experiment_name'],
        run_number=run_number,
        tags=config.MLFLOW_CONFIG_BACKTESTING['tags']
    )
    
    run = exp_manager.start_mlflow_run(run_name_prefix=model_name)
    
    if run.info.status == "FINISHED":
        print(f"🟢 {model_name} run already complete (id={run.info.run_id}) – skipping.")
        return run
    
    try:
        # Log parameters
        if hasattr(forecaster, 'get_params'):
            exp_manager.log_params(forecaster.get_params())
        exp_manager.log_param("model_type", model_name)
        exp_manager.log_param("targets", ",".join(config.TARGETS))
        if extra_params:
            exp_manager.log_params(extra_params)
        
        # Use config horizon if not provided
        if horizon is None:
            horizon = config.HORIZON
        
        print(f"🔄 Training {model_name} with horizon={horizon}...")
        
        # ==================== FORWARD TRANSFORMATIONS ====================
        
        # 1. Scale targets and features
        y_train_scaled = pipe_Y.transform(y_train) if pipe_Y else y_train
        
        X_train_scaled = pipe_X.transform(X_train) if (pipe_X and use_exog) else None
        X_test_scaled = pipe_X.transform(X_test) if (pipe_X and use_exog) else None
        
        # 2. Apply differencing to scaled targets
        if differencer is not None:
            y_train_final = differencer.transform(y_train_scaled)
        else:
            y_train_final = y_train_scaled
            
        # 3. Engineer features
        X_train_final = None
        X_test_final = None
        
        if use_exog and feat_pipeline is not None:
            X_train_final = feat_pipeline.transform(X_train_scaled)
            X_test_final = feat_pipeline.transform(X_test_scaled) 
            
            X_train_final = pd.DataFrame(X_train_final, index=X_train.index)
            X_test_final = pd.DataFrame(X_test_final, index=X_test.index)
        
        # ==================== MODEL TRAINING ====================
        
        if use_exog:
            forecaster.fit(y=y_train_final, X=X_train_final)
        else:
            forecaster.fit(y=y_train_final)
        
        # ==================== BLOCK-WISE PREDICTION ====================
        
        # Create test blocks
        y_test_blocks = create_data_blocks(y_test, horizon)
        n_blocks = len(y_test_blocks)
        
        print(f"  Processing {n_blocks} blocks of size {horizon}")
        exp_manager.log_param("n_blocks", n_blocks)
        exp_manager.log_param("block_size", horizon)
        
        if n_blocks == 0:
            print("⚠️ No complete blocks found. Consider reducing horizon size.")
            return run
        
        # Make predictions for each block
        if recursive_blocks:
            y_pred_blocks_raw = process_blocks_iteratively(
                forecaster, y_train_final, y_test_blocks, X_test_final, use_exog
            )
        else:
            y_pred_blocks_raw = [
                predict_block(forecaster, y_train_final, block, X_test_final, use_exog)
                for block in y_test_blocks
            ]
        
        # ==================== INVERSE TRANSFORMATIONS ====================
        
        y_pred_blocks_final = []
        for y_pred_raw in y_pred_blocks_raw:
            # Inverse transform: undifference -> unscale
            if differencer is not None:
                y_pred_undiffed = differencer.inverse_transform(y_pred_raw)
            else:
                y_pred_undiffed = y_pred_raw
                
            if pipe_Y is not None:
                y_pred_final = pipe_Y.inverse_transform(y_pred_undiffed)
            else:
                y_pred_final = y_pred_undiffed
            
            y_pred_blocks_final.append(y_pred_final)
        
        # ==================== EVALUATION & LOGGING ====================
        
        # Calculate scores for each block
        all_block_scores = []
        if scorers:
            for i, (y_true_block, y_pred_block) in enumerate(zip(y_test_blocks, y_pred_blocks_final)):
                block_scores = compute_block_scores(y_true_block, y_pred_block, scorers)
                all_block_scores.append(block_scores)
                
                # Log individual block scores
                for name, score in block_scores.items():
                    exp_manager.log_metric(f"block_{i+1}_{name}", score)
                    print(f"  Block {i+1} {name}: {score:.4f}")
            
            # Calculate and log aggregated scores
            agg_scores = aggregate_scores(all_block_scores)
            exp_manager.log_metrics(agg_scores)
            
            print(f"\n📊 Aggregated Scores:")
            for name, score in agg_scores.items():
                print(f"  {name}: {score:.4f}")
        
        # Generate plots for each target
        plot_paths = []
        for target in y_test.columns:
            plot_path = plotting.create_multi_block_plot(
                y_test_blocks, y_pred_blocks_final, y_train, y_test,
                model_name, target, run_number, config.REPORTS_DIR
            )
            plot_paths.append(plot_path)
            exp_manager.log_artifact(plot_path, "plots")
        
        print(f"✅ {model_name} training complete ({n_blocks} blocks processed)")
        
    except Exception as e:
        print(f"❌ Error in {model_name}: {str(e)}")
        raise
        
    finally:
        exp_manager.end_mlflow_run()
    
    return run



class PrintMetricsCallback(Callback):
    """
    Custom PyTorch Lightning callback to print training and validation metrics
    at a specified epoch interval.
    """
    def __init__(self, print_every_n_epochs=10):
        self.print_every_n_epochs = print_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Hook that runs at the end of each validation epoch.
        """
        epoch = trainer.current_epoch
        if (epoch) % self.print_every_n_epochs == 0:
            metrics = trainer.callback_metrics
            train_loss = metrics.get('train_loss_epoch')
            val_loss = metrics.get('val_loss')

            print(f"\nEpoch {epoch} Summary:")
            if train_loss is not None:
                print(f"  -> Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  -> Validation Loss: {val_loss:.4f}")
            print("-" * 30)



def evaluate_with_model_metrics(model, val_dataloader):
    """
    Use the model's own loss/metric computation methods.
    This leverages the model's training logic for consistent evaluation.
    """
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            # Use the model's own step method which handles metrics correctly
            if hasattr(model, 'validation_step'):
                step_output = model.validation_step(batch, 0)  # batch_idx=0
                if isinstance(step_output, dict) and 'loss' in step_output:
                    total_loss += step_output['loss'].item()
            
            # Get predictions and targets using model's forward method
            x, y = batch
            if isinstance(y, (list, tuple)):
                targets = y[0]  # Actual targets
            else:
                targets = y
                
            outputs = model(x)
            all_outputs.append(outputs)
            all_targets.append(targets)
    
    # Concatenate all outputs and targets
    if all_outputs:
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
    
    avg_loss = total_loss / len(val_dataloader)
    
    return {
        'predictions': all_outputs,
        'actuals': all_targets,
        'loss': avg_loss
    }

def model_evaluation(predictions, metrics):
    """
    This is a dummy evaluation function that returns dummy errors in a plausible range.
    """
    dummy_metrics = {}
    for metric_name, metric_fn in metrics.items():
        dummy_metrics[f"dummy_{metric_name}"] = np.random.uniform(0.1, 0.5)
    return dummy_metrics


# def evaluate_forecasting_model(best_model, val_loader, metrics):
#     """
#     Evaluate model and compute one or more metrics.

#     Args:
#         best_model: Trained PyTorch Forecasting model
#         val_loader: Validation DataLoader
#         metrics:   A single Metric instance or a list of Metric instances

#     Returns:
#         Tuple of (y_pred, y_true, metrics_dict) where metrics_dict maps
#         metric.name to its computed scalar value.
#     """
#     # ensure we have a list of Metric instances
#     if not isinstance(metrics, (list, tuple)):
#         metrics = [metrics]

#     # get predictions and ground truth
#     pred_val = best_model.predict(
#         val_loader,
#         return_y=True,
#         trainer_kwargs=dict(accelerator="cpu")
#     )
#     # stack into (batch, horizon, n_targets)
#     y_pred = torch.stack(pred_val.output, dim=-1)
#     y_true = torch.stack(pred_val.y[0],   dim=-1)

#     # compute each metric
#     results = {}
#     for metric in metrics:
#         # reset state (if metric has state; most torchmetrics do)
#         try:
#             metric.reset()
#         except AttributeError:
#             pass
#         # update with full batch
#         metric.update(y_pred, y_true)
#         # compute and store as Python float
#         results[metric.name] = metric.compute().item()

#     return y_pred, y_true, results