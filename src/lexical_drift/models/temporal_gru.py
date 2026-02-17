from __future__ import annotations

import torch
from torch import nn


class TemporalGRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _output, hidden = self.gru(x)
        last_hidden = hidden[-1]
        return self.classifier(self.dropout(last_hidden))


def build_temporal_gru(
    input_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
) -> TemporalGRUClassifier:
    return TemporalGRUClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=layers,
        dropout=dropout,
    )
