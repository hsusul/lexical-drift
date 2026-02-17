from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_mlp(input_dim: int, hidden_dim: int, dropout: float) -> MLPClassifier:
    return MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
