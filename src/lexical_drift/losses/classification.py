from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            'PyTorch is required for classification losses. Install with: pip install -e ".[torch]"'
        ) from exc
    return torch


class BinaryFocalLoss:
    def __init__(
        self,
        *,
        gamma: float = 2.0,
        pos_weight: float | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.gamma = float(gamma)
        self.pos_weight = pos_weight
        self.device = device

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        torch = _require_torch()
        F = torch.nn.functional

        pos_weight_tensor = None
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor(
                [float(self.pos_weight)],
                dtype=logits.dtype,
                device=logits.device if self.device is None else self.device,
            )
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=pos_weight_tensor,
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = torch.pow(1.0 - p_t, self.gamma)
        return (focal_factor * bce).mean()


def build_binary_classification_loss(
    *,
    loss_type: str,
    pos_weight: float | None,
    focal_gamma: float,
    device: torch.device,
):
    torch = _require_torch()
    if loss_type == "bce":
        pos_weight_tensor = None
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(
                [float(pos_weight)],
                dtype=torch.float32,
                device=device,
            )
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    if loss_type == "focal":
        return BinaryFocalLoss(gamma=focal_gamma, pos_weight=pos_weight, device=device)
    raise ValueError(f"Unsupported loss_type: {loss_type}")
