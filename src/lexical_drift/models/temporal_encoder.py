from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            'PyTorch is required for temporal encoder features. '
            'Install with: pip install -e ".[torch]"'
        ) from exc
    return torch


def _import_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for e2e temporal runs. "
            'Install with: pip install -e ".[nlp]"'
        ) from exc
    return AutoTokenizer, AutoModel


class TemporalEncoder:
    def __init__(
        self,
        *,
        model_name: str,
        max_length: int,
        pooling: str = "cls",
        freeze: bool = False,
    ) -> None:
        self._torch = _require_torch()
        AutoTokenizer, AutoModel = _import_transformers()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = int(max_length)
        self.pooling = pooling
        self.freeze = bool(freeze)

        if self.freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def to(self, device: torch.device) -> TemporalEncoder:
        self.encoder.to(device)
        return self

    def train(self) -> TemporalEncoder:
        self.encoder.train()
        return self

    def eval(self) -> TemporalEncoder:
        self.encoder.eval()
        return self

    def parameters(self):
        return self.encoder.parameters()

    def state_dict(self):
        return self.encoder.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.encoder.load_state_dict(state_dict, strict=strict)

    @property
    def output_dim(self) -> int:
        return int(self.encoder.config.hidden_size)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            torch = self._torch
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            denom = torch.clamp(mask.sum(dim=1), min=1.0)
            return (hidden * mask).sum(dim=1) / denom
        return hidden[:, 0, :]

    def encode_texts(
        self,
        texts: list[str],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        torch = self._torch
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if self.freeze:
            with torch.no_grad():
                outputs = self.encoder(**encoded)
                hidden = outputs.last_hidden_state
                pooled = self._pool(hidden, encoded["attention_mask"])
            return pooled.detach()
        outputs = self.encoder(**encoded)
        hidden = outputs.last_hidden_state
        return self._pool(hidden, encoded["attention_mask"])

    def encode_sequences(
        self,
        sequences: list[list[str]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        torch = self._torch
        if not sequences:
            return torch.empty((0, 0, self.output_dim), dtype=torch.float32, device=device)
        months = len(sequences[0])
        flat_texts = [text for sequence in sequences for text in sequence]
        pooled = self.encode_texts(flat_texts, device=device)
        return pooled.reshape(len(sequences), months, -1)
