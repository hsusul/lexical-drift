from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from lexical_drift.config import EvalE2EConfig, TrainE2EConfig
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.train import e2e_temporal
from tests._requires_torch import requires_torch


def _hash_to_vector(text: str, dim: int = 32) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    data = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
    if data.size >= dim:
        return data[:dim]
    repeat = int(np.ceil(dim / data.size))
    return np.tile(data, repeat)[:dim]


@requires_torch
def test_e2e_pipeline_smoke(tmp_path, monkeypatch) -> None:
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

        def encode_sequences(
            self,
            sequences: list[list[str]],
            *,
            device: torch.device,
        ) -> torch.Tensor:
            rows: list[np.ndarray] = []
            for sequence in sequences:
                seq_vectors = np.vstack(
                    [_hash_to_vector(text, dim=self._output_dim) for text in sequence]
                )
                rows.append(seq_vectors)
            matrix = np.asarray(rows, dtype=np.float32)
            tensor = torch.from_numpy(matrix).to(device=device, dtype=torch.float32)
            return tensor * self.scale

    monkeypatch.setattr(e2e_temporal, "TemporalEncoder", FakeTemporalEncoder)

    data_path = tmp_path / "synth.csv"
    save_synthetic_dataset(
        out_path=data_path,
        n_authors=20,
        months=6,
        random_seed=5,
        difficulty="hard",
    )

    train_config = TrainE2EConfig(
        input_path=str(data_path),
        output_dir=str(tmp_path / "artifacts"),
        random_seed=5,
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
        pretrained_encoder_path="",
    )
    train_result = e2e_temporal.run_train_e2e(train_config)
    assert Path(str(train_result["model_path"])).exists()
    assert Path(str(train_result["metrics_path"])).exists()
    assert Path(str(train_result["per_month_csv_path"])).exists()
    assert Path(str(train_result["run_metadata_path"])).exists()
    assert Path(str(train_result["plot_paths"]["per_month_metrics_path"])).exists()

    eval_config = EvalE2EConfig(
        input_path=str(data_path),
        output_dir=str(tmp_path / "artifacts"),
        random_seed=5,
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
    assert Path(str(eval_result["model_path"])).exists()
    assert Path(str(eval_result["metrics_path"])).exists()
    assert Path(str(eval_result["per_month_csv_path"])).exists()
    assert Path(str(eval_result["run_metadata_path"])).exists()
    assert Path(str(eval_result["plot_paths"]["per_month_metrics_path"])).exists()
