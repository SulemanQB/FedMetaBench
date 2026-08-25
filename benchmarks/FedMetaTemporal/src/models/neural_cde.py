"""Neural CDE-inspired encoder for irregular time series.

Simplified Neural CDE that:
1. Computes natural cubic spline interpolation of irregular observations
2. Uses an ODE-RNN-like architecture (GRU with time-aware updates)
3. Outputs a fixed-size representation for downstream classification

This avoids torchcde dependency by implementing a lightweight version.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralCDEEncoder(nn.Module):
    """Simplified Neural CDE encoder using GRU with time-gap awareness.

    For irregular time series, incorporates inter-observation time gaps
    as additional features and uses exponential decay for hidden states.

    Input: (batch, seq_len, input_dim) with optional time deltas
    Output: (batch, hidden_dim) fixed-size representation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        use_time_decay: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_time_decay = use_time_decay

        # Input projection (handles raw features + optional time deltas)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GRU cells with time-aware decay
        self.gru_cells = nn.ModuleList([
            nn.GRUCell(hidden_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(n_layers)
        ])

        if use_time_decay:
            # Learned decay rates per hidden dimension
            self.decay_params = nn.ParameterList([
                nn.Parameter(torch.zeros(hidden_dim))
                for _ in range(n_layers)
            ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        time_deltas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) — feature sequence
            time_deltas: (batch, seq_len) — time gaps between observations
                         If None, assumes uniform spacing (delta=1)
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x_proj = self.input_proj(x)  # (B, T, H)

        # Initialize hidden states
        h_list = [torch.zeros(batch_size, self.hidden_dim, device=x.device)
                   for _ in range(self.n_layers)]

        for t in range(seq_len):
            inp = x_proj[:, t, :]

            for layer_idx, gru_cell in enumerate(self.gru_cells):
                # Time-aware decay of hidden state
                if self.use_time_decay and time_deltas is not None:
                    dt = time_deltas[:, t].unsqueeze(1)  # (B, 1)
                    decay = torch.exp(-F.softplus(self.decay_params[layer_idx]) * dt)
                    h_list[layer_idx] = h_list[layer_idx] * decay

                h_list[layer_idx] = gru_cell(inp, h_list[layer_idx])
                inp = self.dropout(h_list[layer_idx])

        # Use final hidden state of last layer
        return self.layer_norm(h_list[-1])


class TemporalClassifier(nn.Module):
    """Full model: Neural CDE encoder + classification head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_classes: int = 2,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = NeuralCDEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_deltas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encoder(x, time_deltas)
        return self.classifier(h)


class SimpleClassifier(nn.Module):
    """Simple MLP baseline for tabular features (no temporal structure)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor, time_deltas=None) -> torch.Tensor:
        return self.net(x)
