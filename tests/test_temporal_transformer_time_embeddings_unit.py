from __future__ import annotations

import pytest

from tests._requires_torch import requires_torch


@requires_torch
def test_temporal_transformer_time_embeddings_affect_forward() -> None:
    torch = pytest.importorskip("torch")
    from lexical_drift.models.temporal_transformer import build_temporal_transformer

    torch.manual_seed(0)

    x = torch.zeros((2, 4, 8), dtype=torch.float32)
    month_indices_a = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    month_indices_b = torch.tensor([[3, 3, 3, 3], [3, 3, 3, 3]], dtype=torch.long)

    model_with_time = build_temporal_transformer(
        input_dim=8,
        hidden_dim=16,
        max_positions=12,
        layers=1,
        heads=4,
        dropout=0.0,
        use_time_embeddings=True,
    )
    model_with_time.eval()
    logits_a = model_with_time(x, month_indices=month_indices_a)
    logits_b = model_with_time(x, month_indices=month_indices_b)
    assert not torch.allclose(logits_a, logits_b)

    model_without_time = build_temporal_transformer(
        input_dim=8,
        hidden_dim=16,
        max_positions=12,
        layers=1,
        heads=4,
        dropout=0.0,
        use_time_embeddings=False,
    )
    model_without_time.eval()
    logits_c = model_without_time(x, month_indices=month_indices_a)
    logits_d = model_without_time(x, month_indices=month_indices_b)
    assert torch.allclose(logits_c, logits_d, atol=1e-6)
