from __future__ import annotations

import torch


class TemporalAttentionClassifier(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        max_positions: int,
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        safe_heads = max(1, int(heads))
        while hidden_dim % safe_heads != 0 and safe_heads > 1:
            safe_heads -= 1

        self.input_projection = torch.nn.Linear(input_dim, hidden_dim)
        self.position_embedding = torch.nn.Embedding(max_positions, hidden_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=safe_heads,
            dim_feedforward=max(hidden_dim * 2, 32),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        positions = torch.arange(sequence_length, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, sequence_length)
        hidden = self.input_projection(x) + self.position_embedding(positions)
        encoded = self.encoder(hidden)
        pooled = encoded.mean(dim=1)
        return self.output(pooled)


def build_temporal_attention(
    *,
    input_dim: int,
    hidden_dim: int,
    max_positions: int,
    layers: int = 1,
    heads: int = 4,
    dropout: float = 0.2,
) -> TemporalAttentionClassifier:
    return TemporalAttentionClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        max_positions=max_positions,
        layers=layers,
        heads=heads,
        dropout=dropout,
    )
