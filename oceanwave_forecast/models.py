import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.view(x.size(0), x.size(1), -1)          # flatten features
        return self.gru(x)                            # (seq, batch, hidden), h_n


class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 hidden_fc: int, output_size: int = 1, num_layers: int = 1):
        super().__init__()
        self.gru  = nn.GRU(input_size, hidden_size, num_layers)
        self.lin1 = nn.Linear(hidden_size, hidden_fc)
        self.lin2 = nn.Linear(hidden_fc, output_size)

    def forward(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        out, h = self.gru(x.unsqueeze(0), h)          # add seq dim
        y = torch.relu(self.lin1(out.squeeze(0)))
        return self.lin2(y), h


class Seq2Seq(nn.Module):
    """Simple encoder-decoder GRU for multi-feature → single-output seq."""

    def __init__(self, hidden_size: int, hidden_fc: int,
                 input_size: int = 3, output_size: int = 1):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(output_size, hidden_size, hidden_fc, output_size)

    def forward(self, src: Tensor, horizon: int,
                teacher_ratio: float = 0.0, trg_truth: Tensor | None = None) -> Tensor:
        
        """
        Forward process for sequence-to-sequence prediction.
        
        The model predicts the next 'horizon' steps using an encoder-decoder architecture:
        
        1. ENCODING PHASE:
           - Process the entire input sequence (src) through the encoder
           - Extract the final hidden state as context for the decoder
        
        2. DECODING PHASE (Autoregressive Generation):
           - For each time step in the prediction horizon:
             a) Feed the current input to the decoder along with hidden state
             b) Decoder outputs: prediction for this step + updated hidden state
             c) Decide next input: either use ground truth (teacher forcing) or own prediction
             d) Repeat for next time step
        
        Args:
            src: Input sequence (seq_len, batch_size, input_features)
            horizon: Number of future steps to predict
            teacher_ratio: Probability of using ground truth vs. model prediction during training
            trg_truth: Ground truth target sequence for teacher forcing (optional)
        
        Returns:
            Predicted sequence (horizon, batch_size, output_features)
        """


        _, h = self.encoder(src)                      # h: (layers, batch, hidden)
        dec_in = src[-1, :, 0].unsqueeze(1)           # latest true value
        outputs = torch.zeros(horizon, src.size(1), 1, device=src.device)

        for t in range(horizon):
            dec_out, h = self.decoder(dec_in, h)
            outputs[t] = dec_out
            use_tf = self.training and trg_truth is not None and torch.rand(1) < teacher_ratio
            dec_in  = trg_truth[t].unsqueeze(1) if use_tf else dec_out
        return outputs

    def predict(self, src: Tensor, horizon: int) -> Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(src, horizon)
