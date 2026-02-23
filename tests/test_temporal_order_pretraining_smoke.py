from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lexical_drift.config import PretrainTemporalOrderConfig
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.train import temporal_order_pretraining
from tests._requires_torch import requires_torch


def _hash_to_vector(text: str, dim: int = 32) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    data = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
    if data.size >= dim:
        return data[:dim]
    repeat = int(np.ceil(dim / data.size))
    return np.tile(data, repeat)[:dim]


@requires_torch
def test_temporal_order_pretraining_smoke(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    class FakeTemporalEncoder(torch.nn.Module):
        def __init__(
            self,
            *,
            model_name: str,
            max_length: int,
            pooling: str = "cls",
            freeze: bool = False,
        ) -> None:
            super().__init__()
            _ = (model_name, max_length, pooling, freeze)
            self.scale = torch.nn.Parameter(torch.ones(1))
            self._output_dim = 32

        @property
        def output_dim(self) -> int:
            return int(self._output_dim)

        def encode_texts(self, texts: list[str], *, device: torch.device) -> torch.Tensor:
            matrix = np.vstack([_hash_to_vector(text, dim=self._output_dim) for text in texts])
            tensor = torch.from_numpy(matrix).to(device=device, dtype=torch.float32)
            return tensor * self.scale

    monkeypatch.setattr(temporal_order_pretraining, "TemporalEncoder", FakeTemporalEncoder)

    data_path = tmp_path / "synth.csv"
    save_synthetic_dataset(
        out_path=data_path,
        n_authors=20,
        months=6,
        random_seed=41,
        difficulty="hard",
    )

    config = PretrainTemporalOrderConfig(
        input_path=str(data_path),
        output_dir=str(tmp_path / "artifacts"),
        random_seed=41,
        encoder_model="fake-encoder",
        max_length=32,
        batch_size=8,
        lr=0.001,
        epochs=1,
        train_months=4,
        hidden_dim=16,
        pooling="cls",
        freeze_encoder=False,
    )

    result = temporal_order_pretraining.run_pretrain_temporal_order(config)
    checkpoint_path = Path(str(result["checkpoint_path"]))
    metrics_path = Path(str(result["metrics_path"]))
    metadata_path = Path(str(result["run_metadata_path"]))

    assert checkpoint_path.exists()
    assert metrics_path.exists()
    assert metadata_path.exists()

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["model_type"] == "temporal_order_pretraining"
    assert int(metrics_payload["n_examples"]) > 0
    assert isinstance(metrics_payload["final_loss"], float)
    assert isinstance(metrics_payload["final_accuracy"], float)
