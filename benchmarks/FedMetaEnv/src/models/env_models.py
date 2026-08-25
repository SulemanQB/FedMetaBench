"""Models for air quality prediction.

EnvLSTM: LSTM-based temporal model for pollutant forecasting.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EnvLSTM(nn.Module):
    """LSTM for air quality time series forecasting.

    Input: (batch, window_size, n_features) — e.g., 24 hours × 17 features
    Output: (batch, 1) — next-step pollutant prediction (regression)
    """

    def __init__(
        self,
        input_dim: int = 17,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(-1)


class EnvMLP(nn.Module):
    """Simple MLP baseline — flattens temporal window."""

    def __init__(
        self,
        input_dim: int = 17,
        window_size: int = 24,
        hidden_dim: int = 128,
    ):
        super().__init__()
        flat_dim = input_dim * window_size
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1)).squeeze(-1)
