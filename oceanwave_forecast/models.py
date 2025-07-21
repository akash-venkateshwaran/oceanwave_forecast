"""
Simplified DeepAR Neural Network for Time Series Forecasting
Production-ready implementation with clean architecture and configuration management.
"""

import math
from typing import Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from loguru import logger


class DeepARNet(nn.Module):
    """
    DeepAR: A recurrent neural network for time series forecasting.
    
    
    """
    
    def __init__(self, config: dict):
        """
        Initialize the DeepAR network.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super(DeepARNet, self).__init__()
        self.config = config
        
        # Validate configuration
        self._validate_config()
        
        # Set device
        self.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Embedding layer for time series IDs
        self.embedding = nn.Embedding(
            config["num_class"], 
            config["embedding_dim"]
        )
        
        # LSTM layer
        lstm_input_size = 1 + config["cov_dim"] + config["embedding_dim"]
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config["lstm_hidden_dim"],
            num_layers=config["lstm_layers"],
            bias=True,
            batch_first=False,
            dropout=config["lstm_dropout"] if config["lstm_layers"] > 1 else 0
        )
        
        # Initialize LSTM forget gate bias to 1 (recommended practice)
        self._init_lstm_weights()
        
        # Output layers for distribution parameters
        hidden_combined_size = config["lstm_hidden_dim"] * config["lstm_layers"]
        self.mu_layer = nn.Linear(hidden_combined_size, 1)
        self.sigma_layer = nn.Sequential(
            nn.Linear(hidden_combined_size, 1),
            nn.Softplus()  # Ensures positive values for sigma
        )
        
        # Move model to device
        self.to(self.device)
        
    def _validate_config(self):
        """Validate configuration parameters."""
        required_keys = [
            "lstm_hidden_dim", "lstm_layers", "lstm_dropout", "embedding_dim",
            "num_class", "cov_dim", "predict_steps", "sample_times", "device"
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
    
    def _init_lstm_weights(self):
        """Initialize LSTM weights, especially forget gate bias to 1."""
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.0)
    
    def forward(self, x: torch.Tensor, idx: torch.Tensor, 
                hidden: torch.Tensor, cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [1, batch_size, 1+cov_dim] - previous value + covariates
            idx: Time series ID [1, batch_size]
            hidden: LSTM hidden state [lstm_layers, batch_size, lstm_hidden_dim]
            cell: LSTM cell state [lstm_layers, batch_size, lstm_hidden_dim]
            
        Returns:
            mu: Predicted mean [batch_size]
            sigma: Predicted standard deviation [batch_size]
            hidden: Updated hidden state
            cell: Updated cell state
        """
        # Get embedding for time series ID
        embed = self.embedding(idx)
        
        # Concatenate input with embedding
        lstm_input = torch.cat((x, embed), dim=2)
        
        # LSTM forward pass
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Combine hidden states from all layers
        hidden_combined = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        
        # Predict distribution parameters
        mu = self.mu_layer(hidden_combined).squeeze()
        sigma = self.sigma_layer(hidden_combined).squeeze()
        
        return mu, sigma, hidden, cell
    
    def init_hidden_state(self, batch_size: int) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(
            self.config["lstm_layers"], 
            batch_size, 
            self.config["lstm_hidden_dim"],
            device=self.device
        )
    
    def init_cell_state(self, batch_size: int) -> torch.Tensor:
        """Initialize cell state."""
        return torch.zeros(
            self.config["lstm_layers"], 
            batch_size, 
            self.config["lstm_hidden_dim"],
            device=self.device
        )
    
    def predict(self, x: torch.Tensor, v_batch: torch.Tensor, 
                id_batch: torch.Tensor, hidden: torch.Tensor, 
                cell: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                                                        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generate predictions for future time steps.
        
        Args:
            x: Input sequence
            v_batch: Scaling parameters (scale, shift)
            id_batch: Time series IDs
            hidden: Initial hidden state
            cell: Initial cell state
            probabilistic: Whether to return probabilistic samples
            
        Returns:
            If probabilistic=False: (predictions, uncertainties)
            If probabilistic=True: (samples, predictions, uncertainties)
        """
        self.eval()
        batch_size = x.shape[1]
        predict_steps = self.config["predict_steps"]
        predict_start = self.config.get("predict_start", x.shape[0] - predict_steps)
        
        with torch.no_grad():
                return self._point_predict(x, v_batch, id_batch, hidden, cell,
                                         predict_start, predict_steps, batch_size)
    
        return samples, sample_mean, sample_std
    
    def _point_predict(self, x, v_batch, id_batch, hidden, cell,
                      predict_start, predict_steps, batch_size):
        """Generate point predictions."""
        decoder_hidden = hidden
        decoder_cell = cell
        predictions = torch.zeros(batch_size, predict_steps, device=self.device)
        uncertainties = torch.zeros(batch_size, predict_steps, device=self.device)
        
        for t in range(predict_steps):
            mu, sigma, decoder_hidden, decoder_cell = self(
                x[predict_start + t].unsqueeze(0),
                id_batch, decoder_hidden, decoder_cell
            )
            
            # Store scaled predictions
            predictions[:, t] = mu * v_batch[:, 0] + v_batch[:, 1]
            uncertainties[:, t] = sigma * v_batch[:, 0]
            
            # Update input for next step
            if t < predict_steps - 1:
                x[predict_start + t + 1, :, 0] = mu
        
        return predictions, uncertainties


def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, 
                      targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute Gaussian negative log-likelihood loss.
    
    Args:
        mu: Predicted means [batch_size]
        sigma: Predicted standard deviations [batch_size]
        targets: Target values [batch_size]
        mask: Optional mask for valid values [batch_size]
        
    Returns:
        Average negative log-likelihood loss
    """
    if mask is None:
        mask = (targets != 0)
    
    if not mask.any():
        return torch.tensor(0.0, device=mu.device, requires_grad=True)
    
    mu_masked = mu[mask]
    sigma_masked = sigma[mask]
    targets_masked = targets[mask]
    
    # Gaussian negative log-likelihood
    nll = 0.5 * torch.log(2 * math.pi * sigma_masked.pow(2)) + \
          0.5 * ((targets_masked - mu_masked) / sigma_masked).pow(2)
    
    return nll.mean()


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None) -> dict:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Model predictions [batch_size, seq_len] or [batch_size]
        targets: Target values [batch_size, seq_len] or [batch_size]
        mask: Optional mask for valid values
        
    Returns:
        Dictionary containing computed metrics
    """
    if mask is None:
        mask = (targets != 0)
    
    if not mask.any():
        return {"mae": 0.0, "rmse": 0.0, "nd": 0.0}
    
    pred_masked = predictions[mask]
    target_masked = targets[mask]
    
    # Mean Absolute Error
    mae = torch.abs(pred_masked - target_masked).mean().item()
    
    # Root Mean Square Error
    rmse = torch.sqrt(torch.pow(pred_masked - target_masked, 2).mean()).item()
    
    # Normalized Deviation
    nd = torch.abs(pred_masked - target_masked).sum().item() / torch.abs(target_masked).sum().item()
    
    return {
        "mae": mae,
        "rmse": rmse,
        "nd": nd
    }