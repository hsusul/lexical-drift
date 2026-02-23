from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from lexical_drift.config import EvalE2EConfig, PretrainContrastiveConfig, TrainE2EConfig
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.train import contrastive_temporal, e2e_temporal


def _hash_to_vector(text: str, dim: int = 32) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    data = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
    if data.size >= dim:
        return data[:dim]
    repeat = int(np.ceil(dim / data.size))
    return np.tile(data, repeat)[:dim]


def test_contrastive_pretrain_and_downstream_load(tmp_path, monkeypatch) -> None:
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
            rows = np.vstack(
                [_hash_to_vector(text, dim=self._output_dim) for text in texts]
            ).astype(np.float32)
            return torch.from_numpy(rows).to(device=device) * self.scale

        def encode_sequences(
            self,
            sequences: list[list[str]],
            *,
            device: torch.device,
        ) -> torch.Tensor:
            rows: list[np.ndarray] = []
            for sequence in sequences:
                rows.append(
                    np.vstack([_hash_to_vector(text, dim=self._output_dim) for text in sequence])
                )
            matrix = np.asarray(rows, dtype=np.float32)
            return torch.from_numpy(matrix).to(device=device) * self.scale

    monkeypatch.setattr(contrastive_temporal, "TemporalEncoder", FakeTemporalEncoder)
    monkeypatch.setattr(e2e_temporal, "TemporalEncoder", FakeTemporalEncoder)

    data_path = tmp_path / "synth.csv"
    save_synthetic_dataset(
        out_path=data_path,
        n_authors=20,
        months=6,
        random_seed=9,
        difficulty="hard",
    )

    pretrain_config = PretrainContrastiveConfig(
        input_path=str(data_path),
        output_dir=str(tmp_path / "artifacts"),
        random_seed=9,
        encoder_model="fake-encoder",
        max_length=32,
        batch_size=4,
        lr=0.001,
        epochs=1,
        temperature=0.2,
        projection_dim=16,
        train_months=3,
        pooling="cls",
        freeze_encoder=False,
    )
    pretrain_result = contrastive_temporal.run_pretrain_contrastive(pretrain_config)
    checkpoint_path = Path(str(pretrain_result["checkpoint_path"]))
    assert checkpoint_path.exists()

    train_config = TrainE2EConfig(
        input_path=str(data_path),
        output_dir=str(tmp_path / "artifacts"),
        random_seed=9,
        encoder_model="fake-encoder",
        max_length=32,
        batch_size=4,
        train_months=3,
        gru_hidden_dim=16,
        gru_layers=1,
        dropout=0.1,
        lr=0.001,
        epochs=1,
        test_size=0.25,
        pooling="cls",
        freeze_encoder=False,
        pretrained_encoder_path=str(checkpoint_path),
    )
    train_result = e2e_temporal.run_train_e2e(train_config)
    assert Path(str(train_result["model_path"])).exists()

    eval_config = EvalE2EConfig(
        input_path=str(data_path),
        output_dir=str(tmp_path / "artifacts"),
        random_seed=9,
        encoder_model="fake-encoder",
        max_length=32,
        batch_size=4,
        train_months=3,
        checkpoint_path=str(train_result["model_path"]),
        test_size=0.25,
        pooling="cls",
        threshold=0.5,
    )
    eval_result = e2e_temporal.run_eval_e2e(eval_config)
    assert Path(str(eval_result["metrics_path"])).exists()
