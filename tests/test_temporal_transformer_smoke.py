from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from lexical_drift.config import EvalTemporalConfig
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.eval import eval_temporal


def _fake_encode_texts_to_embeddings(
    texts: list[str],
    model_name: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    _ = batch_size
    rows: list[np.ndarray] = []
    for text in texts:
        digest = hashlib.sha256(f"{model_name}|{max_length}|{text}".encode()).digest()
        vector = np.frombuffer(digest, dtype=np.uint8)[:32].astype(np.float32) / 255.0
        rows.append(vector)
    return np.vstack(rows)


def test_temporal_transformer_eval_smoke(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    data_path = tmp_path / "synth_transformer.csv"
    save_synthetic_dataset(
        data_path,
        n_authors=20,
        months=6,
        random_seed=31,
        difficulty="hard",
    )
    monkeypatch.setattr(
        eval_temporal,
        "encode_texts_to_embeddings",
        _fake_encode_texts_to_embeddings,
    )

    config = EvalTemporalConfig(
        input_path=str(data_path),
        output_dir=str(tmp_path / "artifacts"),
        random_seed=31,
        model_type="transformer",
        encoder_model="distilbert-base-uncased",
        max_length=64,
        batch_size=8,
        cache_embeddings=True,
        cache_dir=str(tmp_path / "cache"),
        train_months=3,
        gru_hidden_dim=32,
        gru_layers=1,
        dropout=0.1,
        lr=0.001,
        epochs=1,
        threshold_mode="fixed",
        fixed_threshold=0.5,
        calibration_metric="balanced_accuracy",
        test_size=0.2,
    )

    result = eval_temporal.run_eval_temporal(config)
    assert result["model_type"] == "transformer"
    assert Path(str(result["metrics_path"])).exists()
    assert Path(str(result["model_path"])).exists()
    assert Path(str(result["per_month_csv_path"])).exists()
    plot_paths = dict(result["plot_paths"])
    assert Path(str(plot_paths["per_month_metrics_path"])).exists()
    assert Path(str(plot_paths["attention_over_time_path"])).exists()
