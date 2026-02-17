from __future__ import annotations

import pytest


def test_temporal_gru_forward_shape() -> None:
    torch = pytest.importorskip("torch")

    from lexical_drift.models.temporal_gru import build_temporal_gru

    batch = 4
    months = 6
    input_dim = 32
    x = torch.randn(batch, months, input_dim)

    model = build_temporal_gru(
        input_dim=input_dim,
        hidden_dim=16,
        layers=1,
        dropout=0.2,
    )
    logits = model(x)
    assert tuple(logits.shape) == (batch, 1)
