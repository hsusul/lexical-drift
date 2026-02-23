from __future__ import annotations

import torch
from torch import nn


class _TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        safe_heads = max(1, int(heads))
        while hidden_dim % safe_heads != 0 and safe_heads > 1:
            safe_heads -= 1
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=safe_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim * 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim * 2, 32), hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights


class TemporalTransformer(nn.Module):
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
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = nn.Embedding(max_positions, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                _TemporalTransformerBlock(hidden_dim=hidden_dim, heads=heads, dropout=dropout)
                for _ in range(max(1, int(layers)))
            ]
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size, sequence_length, _ = x.shape
        positions = torch.arange(sequence_length, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, sequence_length)
        hidden = self.input_projection(x) + self.position_embedding(positions)

        attention_weights: list[torch.Tensor] = []
        for block in self.blocks:
            hidden, weights = block(hidden)
            attention_weights.append(weights)

        pooled = hidden.mean(dim=1)
        logits = self.output(pooled)
        if return_attention:
            return logits, attention_weights
        return logits


def build_temporal_transformer(
    *,
    input_dim: int,
    hidden_dim: int,
    max_positions: int,
    layers: int = 1,
    heads: int = 4,
    dropout: float = 0.2,
) -> TemporalTransformer:
    return TemporalTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        max_positions=max_positions,
        layers=layers,
        heads=heads,
        dropout=dropout,
    )
