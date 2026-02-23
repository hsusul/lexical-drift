from __future__ import annotations

import torch
from torch.nn import functional as F


def info_nce_loss(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if anchor_embeddings.ndim != 2 or positive_embeddings.ndim != 2:
        raise ValueError("anchor_embeddings and positive_embeddings must be 2D tensors")
    if anchor_embeddings.shape != positive_embeddings.shape:
        raise ValueError("anchor_embeddings and positive_embeddings must have equal shapes")
    if anchor_embeddings.shape[0] < 2:
        raise ValueError("batch size must be >= 2 for in-batch negatives")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    z_anchor = F.normalize(anchor_embeddings, dim=1)
    z_positive = F.normalize(positive_embeddings, dim=1)
    logits = (z_anchor @ z_positive.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_ab + loss_ba)
