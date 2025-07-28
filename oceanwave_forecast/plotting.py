from typing import Dict, Tuple, Optional
import pandas as pd
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from typing import List
import torch
from torch.utils.data import DataLoader
import numpy as np

def _stack_targets(t, sample_idx: int, length: int) -> torch.Tensor:
    """
    Return shape (length, n_targets). Works whether `t` is a tensor or list[tensor].
    For encoder_target/decoder_target in x_batch or y_batch[0].
    """
    if isinstance(t, list):
        # each element: [batch, time] or [batch, time, 1]
        parts = []
        for tt in t:
            if tt.ndim == 2:
                parts.append(tt[sample_idx, :length].unsqueeze(-1))
            elif tt.ndim == 3:
                parts.append(tt[sample_idx, :length, :])
            else:
                raise ValueError(f"Unexpected ndim {tt.ndim} for target element.")
        return torch.cat(parts, dim=-1)  # [length, n_targets]
    else:
        # tensor
        if t.ndim == 2:                  # [batch, time]
            return t[sample_idx, :length].unsqueeze(-1)
        elif t.ndim == 3:                # [batch, time, n_targets]
            return t[sample_idx, :length, :]
        else:
            raise ValueError(f"Unexpected ndim {t.ndim} for target tensor.")


def plot_dataloader_sample(dataloader: DataLoader, output_path: Path) -> None:
    x_batch, y_batch = next(iter(dataloader))

    # map batch to original dataframe indices
    batch_index_df = dataloader.dataset.x_to_index(x_batch)
    unique_group_ids_in_batch = batch_index_df["group_id"].unique()

    # get target names in the same order as dataset.target (string or list[str])
    target_names = dataloader.dataset.target
    if isinstance(target_names, str):
        target_names = [target_names]
    n_targets = len(target_names)

    n_samples_to_plot = len(unique_group_ids_in_batch)
    fig, axes = plt.subplots(
        n_samples_to_plot, 2, figsize=(20, 6 * n_samples_to_plot), constrained_layout=True
    )
    if n_samples_to_plot == 1:
        axes = np.array([axes])

    for plot_idx, group_id_to_plot in enumerate(unique_group_ids_in_batch):
        sample_in_batch_idx = batch_index_df[batch_index_df["group_id"] == group_id_to_plot].index[0]

        enc_len = x_batch["encoder_lengths"][sample_in_batch_idx].item()
        dec_len = x_batch["decoder_lengths"][sample_in_batch_idx].item()

        ax_x = axes[plot_idx, 0]
        ax_y = axes[plot_idx, 1]

        # time indices
        first_decoder_t = x_batch["decoder_time_idx"][sample_in_batch_idx, 0].item()
        encoder_time_idx = torch.arange(first_decoder_t - enc_len, first_decoder_t).cpu().numpy()
        decoder_time_idx = x_batch["decoder_time_idx"][sample_in_batch_idx, :dec_len].cpu().numpy()
        full_time_idx = np.concatenate([encoder_time_idx, decoder_time_idx])

        # ----- targets from x (scaled) -----
        enc_t = _stack_targets(x_batch["encoder_target"], sample_in_batch_idx, enc_len).cpu().numpy()
        dec_t_x = _stack_targets(x_batch["decoder_target"], sample_in_batch_idx, dec_len).cpu().numpy()
        full_t = np.concatenate([enc_t, dec_t_x], axis=0)  # [enc_len+dec_len, n_targets]

        for i_t, name in enumerate(target_names):
            ax_x.plot(full_time_idx[:enc_len], full_t[:enc_len, i_t], marker="o", linestyle="-",
                      label=f"{name} encoder (scaled)")
            ax_x.plot(full_time_idx[enc_len:], full_t[enc_len:, i_t], marker="x", linestyle="--",
                      label=f"{name} decoder (scaled)")

        # ----- known real covariates (scaled) -----
        known_reals = [c for c in dataloader.dataset.time_varying_known_reals if c != "time_idx"]
        for col_name in known_reals:
            if col_name in dataloader.dataset.reals:
                idx_in_reals = dataloader.dataset.reals.index(col_name)
                enc_cov = x_batch["encoder_cont"][sample_in_batch_idx, :enc_len, idx_in_reals].cpu().numpy()
                dec_cov = x_batch["decoder_cont"][sample_in_batch_idx, :dec_len, idx_in_reals].cpu().numpy()
                ax_x.plot(full_time_idx, np.concatenate([enc_cov, dec_cov]),
                          linestyle=":", label=f"{col_name} (scaled)")

        ax_x.set_title(f"Group {group_id_to_plot} – scaled encoder/decoder + covariates")
        ax_x.set_xlabel("time_idx")
        ax_x.set_ylabel("value")
        ax_x.grid(True, alpha=0.3)
        ax_x.legend(loc="best")

        # ----- actual target from y -----
        y_dec = y_batch[0]  # first element is always the target(s)
        y_dec_t = _stack_targets(y_dec, sample_in_batch_idx, dec_len).cpu().numpy()  # [dec_len, n_targets]
        for i_t, name in enumerate(target_names):
            ax_y.plot(decoder_time_idx, y_dec_t[:, i_t], marker="o", linestyle="-",
                      label=f"{name} (actual)")
        ax_y.set_title(f"Group {group_id_to_plot} – decoder ground truth (y)")
        ax_y.set_xlabel("time_idx")
        ax_y.set_ylabel("value")
        ax_y.grid(True, alpha=0.3)
        ax_y.legend(loc="best")

    plt.suptitle("Sample batch visualization", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Sample batch data visualization saved to {output_path}")



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




def plot_predictions_with_loss_overlay(best_model, val_loader, trainer):
    """
    Plot predictions with training/validation loss overlay on secondary y-axis.
    
    Args:
        best_model: Trained model
        val_loader: Validation data loader
        trainer: Lightning trainer (contains training history)
    """
    # Get raw predictions for plotting
    raw_pred = best_model.predict(
        val_loader,
        mode="raw",
        return_x=True,
        trainer_kwargs=dict(accelerator="cpu")
    )

    # Create the prediction plot
    fig = best_model.plot_prediction(
        raw_pred.x,
        raw_pred.output,
        add_loss_to_title=True
    )
    
    # Add loss overlay on secondary y-axis
    if hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'get_scalars'):
        try:
            # Try to get training history (this depends on the logger used)
            # This is a simplified approach - actual implementation may vary based on your logger
            ax_main = plt.gca()
            ax_loss = ax_main.twinx()
            
            # If you have access to training history, plot it here
            # Example (you may need to adapt this based on your logger):
            # epochs = range(len(train_losses))
            # ax_loss.plot(epochs, train_losses, 'r--', alpha=0.7, label='Train Loss')
            # ax_loss.plot(epochs, val_losses, 'b--', alpha=0.7, label='Val Loss')
            # ax_loss.set_ylabel('Loss', color='red')
            # ax_loss.legend(loc='upper right')
            
            # Placeholder for demonstration - replace with actual loss data
            ax_loss.set_ylabel('Loss (Secondary Axis)', color='red')
            ax_loss.tick_params(axis='y', labelcolor='red')
            
        except Exception as e:
            print(f"Could not add loss overlay: {e}")
    
    plt.tight_layout()
    return fig

def model_plotting_function(model, test_loader):
    """
    This is a dummy plotting function that does nothing and returns None.
    """
    return None
