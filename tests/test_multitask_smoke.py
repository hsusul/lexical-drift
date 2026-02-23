from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lexical_drift.config import TrainMultiTaskConfig
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.train import multitask_temporal


def _hash_to_vector(text: str, dim: int = 32) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    data = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
    if data.size >= dim:
        return data[:dim]
    repeat = int(np.ceil(dim / data.size))
    return np.tile(data, repeat)[:dim]


def test_multitask_train_and_ablation_smoke(tmp_path, monkeypatch) -> None:
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
                rows.append(
                    np.vstack([_hash_to_vector(text, dim=self._output_dim) for text in sequence])
                )
            matrix = np.asarray(rows, dtype=np.float32)
            return torch.from_numpy(matrix).to(device=device) * self.scale

    monkeypatch.setattr(multitask_temporal, "TemporalEncoder", FakeTemporalEncoder)

    data_path = tmp_path / "synth.csv"
    save_synthetic_dataset(
        out_path=data_path,
        n_authors=20,
        months=6,
        random_seed=15,
        difficulty="hard",
    )

    config = TrainMultiTaskConfig(
        input_path=str(data_path),
        output_dir=str(tmp_path / "artifacts"),
        random_seed=15,
        encoder_model="fake-encoder",
        max_length=32,
        batch_size=4,
        train_months=3,
        hidden_dim=16,
        layers=1,
        dropout=0.1,
        lr=0.001,
        epochs=1,
        test_size=0.25,
        drift_lambda=0.3,
        drift_target_metric="cosine",
        pooling="cls",
        freeze_encoder=False,
        threshold=0.5,
    )
    train_result = multitask_temporal.run_train_multitask(config)
    assert Path(str(train_result["model_path"])).exists()
    assert Path(str(train_result["metrics_path"])).exists()
    assert Path(str(train_result["per_month_csv_path"])).exists()
    assert Path(str(train_result["run_metadata_path"])).exists()

    ablation = multitask_temporal.run_ablation_drift_weight(
        config_template=config,
        lambdas=[0.0, 0.3],
        seeds=[1, 2],
        n_authors=20,
        months=6,
        difficulty="hard",
        artifact_root=tmp_path / "ablation",
    )
    summary_path = Path(str(ablation["summary_path"]))
    plot_path = Path(str(ablation["plot_path"]))
    assert summary_path.exists()
    assert plot_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["lambdas"] == [0.0, 0.3]
    assert payload["seeds"] == [1, 2]
    assert isinstance(payload["rows"], list)
    assert len(payload["rows"]) == 2
