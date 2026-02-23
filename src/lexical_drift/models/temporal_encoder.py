from __future__ import annotations

import torch
from torch import nn


def _import_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            'transformers is required for e2e temporal runs. Install with: pip install -e ".[nlp]"'
        ) from exc
    return AutoTokenizer, AutoModel


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        max_length: int,
        pooling: str = "cls",
        freeze: bool = False,
    ) -> None:
        super().__init__()
        AutoTokenizer, AutoModel = _import_transformers()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = int(max_length)
        self.pooling = pooling
        self.freeze = bool(freeze)

        if self.freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    @property
    def output_dim(self) -> int:
        return int(self.encoder.config.hidden_size)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
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
        if not sequences:
            return torch.empty((0, 0, self.output_dim), dtype=torch.float32, device=device)
        months = len(sequences[0])
        flat_texts = [text for sequence in sequences for text in sequence]
        pooled = self.encode_texts(flat_texts, device=device)
        return pooled.reshape(len(sequences), months, -1)
