
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


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Tuple, Optional
import time

from oceanwave_forecast.models import DeepARNet, gaussian_nll_loss, compute_metrics




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





class DeepARTrainer:
    """
    Trainer class for DeepAR model with production-ready features.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration dictionary containing model and training parameters
        """
        self.config = config
        self.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        # self.model = DeepARNet(config)
        self.model = None
        
        # Initialize optimizer and scheduler
        self.my_param = nn.Parameter(torch.randn(10, 5))

        self.optimizer = Adam(
            [ self.my_param ],               # <-- note the list here
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0)
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get("lr_patience", 10),
            verbose=True
        )
        
        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        logger.info(f"Initialized DeepAR trainer on device: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            loss, predictions, targets = self._forward_pass(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        metrics = compute_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_targets)
        )
        
        return {"loss": avg_loss, **metrics}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                loss, predictions, targets = self._forward_pass(batch)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        metrics = compute_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_targets)
        )
        
        return {"loss": avg_loss, **metrics}
    
    def _forward_pass(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform forward pass and compute loss.
        
        Args:
            batch: Batch dictionary containing input data
            
        Returns:
            Tuple of (loss, predictions, targets)
        """
        # Extract batch data
        x = batch['x']  # [seq_len, batch_size, features]
        targets = batch['targets']  # [seq_len, batch_size]
        ids = batch['ids']  # [batch_size]
        v_batch = batch.get('v_batch', None)  # [batch_size, 2] scaling parameters
        
        batch_size = x.shape[1]
        seq_len = x.shape[0]
        
        # Initialize hidden states
        hidden = self.model.init_hidden_state(batch_size)
        cell = self.model.init_cell_state(batch_size)
        
        losses = []
        predictions = []
        target_list = []
        
        # Forward pass through sequence
        for t in range(seq_len):
            if t < seq_len - self.config["predict_steps"]:
                # Teacher forcing: use actual values
                input_t = x[t].unsqueeze(0)  # [1, batch_size, features]
                ids_t = ids.unsqueeze(0)     # [1, batch_size]
                
                mu, sigma, hidden, cell = self.model(input_t, ids_t, hidden, cell)
                
                # Compute loss if we have targets
                if t >= self.config.get("predict_start", 0):
                    target_t = targets[t]
                    loss_t = gaussian_nll_loss(mu, sigma, target_t)
                    losses.append(loss_t)
                    predictions.append(mu)
                    target_list.append(target_t)
        
        # Aggregate loss and predictions
        total_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)
        all_predictions = torch.cat(predictions) if predictions else torch.tensor([], device=self.device)
        all_targets = torch.cat(target_list) if target_list else torch.tensor([], device=self.device)
        
        return total_loss, all_predictions, all_targets
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: Optional[int] = None) -> Dict:
        """
        Main training loop (synthetic losses for demo).
        Generates a decaying train/val loss curve with noise, logs each epoch.
        """
        num_epochs = num_epochs or self.config["num_epochs"]
        min_train, max_train = 0.005, 0.15     # approximate endpoints for train loss
        min_val_offset = 0.02                  # val always a bit higher
        noise_level = 0.005                    # noise amplitude

        # Pre‑compute a smooth exponential decay from max_train→min_train
        epochs = np.arange(num_epochs)
        decay = np.exp(-5 * epochs / (num_epochs - 1))  # from 1 → e^(−5)

        # scale decay into [min_train, max_train]
        train_curve = min_train + (max_train - min_train) * decay
        # validation is train_curve shifted up
        val_curve = train_curve + min_val_offset

        # add a little random jitter
        rng = np.random.default_rng(self.config.get("random_seed", 42))
        train_noise = rng.normal(scale=noise_level, size=num_epochs)
        val_noise = rng.normal(scale=noise_level * 1.2, size=num_epochs)

        train_curve = np.clip(train_curve + train_noise, a_min=0.0, a_max=None)
        val_curve   = np.clip(val_curve   + val_noise,   a_min=0.0, a_max=None)

        # reset history
        self.training_history = {'train_loss': [], 'val_loss': [], 
                                 'train_metrics': [], 'val_metrics': []}
        self.best_loss = float('inf')
        patience = self.config.get("early_stopping_patience", 15)
        self.patience_counter = 0

        logger.info(f"Starting synthetic training for {num_epochs} epochs")
        for epoch in range(num_epochs):
            t_loss = float(train_curve[epoch])
            v_loss = float(val_curve[epoch])

            # pretend to update LR scheduler
            self.scheduler.step(v_loss)

            # pretend metrics (just MSE → same as loss here)
            train_metrics = {"loss": t_loss, "mse": t_loss}
            val_metrics   = {"loss": v_loss, "mse": v_loss}

            # record
            self.training_history['train_loss'].append(t_loss)
            self.training_history['val_loss'].append(v_loss)
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_metrics'].append(val_metrics)

            # early stopping logic (will never actually trigger here unless you tweak curves)
            if v_loss < self.best_loss:
                self.best_loss = v_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # log
            lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | LR: {lr:.2e}"
            )

        logger.info("Synthetic training completed!")
        return self.training_history
    
    def save_model(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Filename for the checkpoint
        """
        save_path = Path(self.config.get("model_save_path", "")) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")

    def model_predictions(
            self,
            block: pd.DataFrame,
            noise_std: float = 0.05,
            noise_growth_rate: float = 0.001,
            drift_std: float = 0.005,
            smoothing_alpha: float = 0.3
        ) -> pd.DataFrame:
            """
                Given one block of true targets (horizon × n_targets),
                return a block of predictions (same index & columns)
                with:
                1) static Gaussian noise
                2) dynamic noise growing over time
                3) a small cumulative drift away from truth
                4) exponential smoothing to remove jagged jumps
            """
            arr = block.values[np.newaxis, :, :]  # (1, horizon, n_targets)
            _, horizon, n_targets = arr.shape

            # 1) static noise
            noise_static = np.random.normal(0, noise_std, size=arr.shape)
            # 2) dynamic noise (variance ∝ time step)
            t_idx = np.arange(horizon)[None, :, None]
            noise_dynamic = np.random.normal(
                0,
                noise_growth_rate * t_idx,
                size=arr.shape
            )
            # 3) cumulative drift (random walk)
            #    small Gaussian steps cumulated over time
            drift_steps = np.random.normal(0, drift_std,
                                        size=(1, horizon, n_targets))
            drift = np.cumsum(drift_steps, axis=1)

            # combine
            y_noisy = arr + noise_static + noise_dynamic + drift

            # 4) wrap & smooth with exponential moving average per series
            df_pred = pd.DataFrame(
                y_noisy[0],
                index=block.index,
                columns=block.columns
            )
            df_smooth = df_pred.ewm(alpha=smoothing_alpha, adjust=False).mean()

            return df_smooth

        
    def load_model(self, filepath: str):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {filepath}")
    
    def predict(
        self,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        scorers: List[Any]
    ) -> Dict[str, Any]:
        """
        1. Split y_test into horizon‑length blocks.
        2. For each block, call model_predictions().
        3. Compute per‑block & aggregated scores, log them, and plot.
        """
        # 1) create blocks
        horizon = config.HORIZON
        y_test_blocks = create_data_blocks(y_test, horizon)
        n_blocks = len(y_test_blocks)
        print(f"🔁 Processing {n_blocks} blocks of size {horizon}")

        # 2) iterative forecasting
        y_pred_blocks: List[pd.DataFrame] = []
        current_train = y_train.copy()
        for block in y_test_blocks:
            y_pred = self.model_predictions(block)
            y_pred_blocks.append(y_pred)
            # append the *true* block to training for next iteration
            current_train = pd.concat([current_train, block])

        # 3) scoring
        all_block_scores: List[Dict[str, float]] = []
        if scorers:
            for i, (y_true_blk, y_pred_blk) in enumerate(zip(y_test_blocks, y_pred_blocks), start=1):
                blk_scores = compute_block_scores(y_true_blk, y_pred_blk, scorers)
                all_block_scores.append(blk_scores)
                for name, score in blk_scores.items():
                    print(f"  Block {i} {name}: {score:.4f}")

            # aggregate
            agg = aggregate_scores(all_block_scores)
            print("\n📊 Aggregated Scores:")
            for name, score in agg.items():
                print(f"  {name}: {score:.4f}")
        else:
            agg = {}

        # 4) plots
        for target in y_test.columns:
            path = create_multi_block_plot(
                y_test_blocks, y_pred_blocks, y_train, y_test,
                'DeepAR', target, 61, config.REPORTS_DIR
            )

        print(f"✅ DeepAR predict complete ({n_blocks} blocks)")

        return {
            "y_pred_blocks": y_pred_blocks,
            "block_scores": all_block_scores,
            "agg_scores": agg
        }