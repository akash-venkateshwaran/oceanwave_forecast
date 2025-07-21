
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
import matplotlib.pyplot as plt
from sktime.forecasting.base import ForecastingHorizon
from oceanwave_forecast import config, mlflow_utils
import pandas as pd
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sktime.forecasting.base import ForecastingHorizon
from typing import List, Dict, Optional, Tuple, Any



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

def create_multi_block_plot(y_true_blocks: List[pd.DataFrame], 
                           y_pred_blocks: List[pd.DataFrame],
                           y_train: pd.DataFrame, 
                           y_test: pd.DataFrame,
                           model_name: str, 
                           target: str, 
                           run_number: int, 
                           save_dir: Path) -> str:
    """Create vertically stacked subplots for each block showing predictions vs truth."""
    n_blocks = len(y_true_blocks)
    
    # Create full time series for context calculation
    y_full = pd.concat([y_train, y_test])
    
    # Create vertically stacked subplots
    fig, axes = plt.subplots(n_blocks, 1, figsize=(12, 4 * n_blocks))
    if n_blocks == 1:
        axes = [axes]
    
    for i, (y_true_block, y_pred_block) in enumerate(zip(y_true_blocks, y_pred_blocks)):
        ax = axes[i]
        
        # Find the context: 50 points before the start of current block
        block_start_idx = y_full.index.get_loc(y_true_block.index[0])
        context_start_idx = max(0, block_start_idx - 50)
        context_data = y_full[target].iloc[context_start_idx:block_start_idx]
        
        # Plot context if available
        if len(context_data) > 0:
            ax.plot(context_data.index, context_data.values, 
                    label='Context', color='lightblue', alpha=0.7)
        
        # Plot truth and predictions
        ax.plot(y_true_block.index, y_true_block[target], 
                label='Truth', color='blue', linewidth=2)
        ax.plot(y_pred_block.index, y_pred_block[target], 
                label='Prediction', color='red', linestyle='--', linewidth=2)
        
        ax.set_title(f'Block {i+1}: {target}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'{model_name} - {target} (Blocks: {n_blocks})', fontsize=14)
    plt.tight_layout()
    
    plot_path = save_dir / f"Run{run_number}_{model_name}_{target}_blocks.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return str(plot_path)

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
            plot_path = create_multi_block_plot(
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
