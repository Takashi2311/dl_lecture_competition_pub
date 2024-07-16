import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torchvision import models


class BasicTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        num_head: int = 8,
        num_layers: int = 4,
        dropout: float = 0.3
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, hid_dim, kernel_size=3, padding=1)
        self.pos_encoder = Position(hid_dim, len=seq_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=num_head, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hid_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),  # ドロップアウトで正則化
            nn.Linear(256, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.conv1(X)
        X = X.permute(2, 0, 1)
        X = self.pos_encoder(X)
        eeg_features = self.transformer_encoder(X)
        eeg_features = eeg_features.mean(dim=0)

        return self.head(eeg_features)

class Position(nn.Module):
    def __init__(self, d_model, len=10000):
        super().__init__()
        pe = torch.zeros(len, d_model)
        position = torch.arange(0, len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encodings added, shape (batch_size, seq_len, d_model)
        """
        # Add positional encoding to input tensor
        x = x + self.pe[:, :x.size(1)]
        return x
