"""Clinical models for eICU temporal data.

ClinicalLSTM: LSTM-based model for ICU mortality prediction from temporal features.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClinicalLSTM(nn.Module):
    """LSTM model for temporal clinical feature sequences.

    Input: (batch, seq_len, input_dim) — e.g., 48 hourly timesteps × D features
    Output: (batch, n_classes) — mortality prediction logits
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        fc_in = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        out, (h_n, _) = self.lstm(x)
        # Use last hidden state
        if self.lstm.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last = h_n[-1]
        return self.fc(self.dropout(last))


class ClinicalGRU(nn.Module):
    """GRU variant for clinical temporal data."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h_n = self.gru(x)
        return self.fc(self.dropout(h_n[-1]))
