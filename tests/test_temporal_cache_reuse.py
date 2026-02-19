from __future__ import annotations

import hashlib

import numpy as np
import pytest

from lexical_drift.config import TemporalTrainConfig
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.training import train_temporal


def _fake_encode_texts_to_embeddings(
    texts: list[str],
    model_name: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    _ = (model_name, max_length, batch_size)
    rows: list[np.ndarray] = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(digest[:32], dtype=np.uint8).astype(np.float32) / 255.0
        rows.append(vec)
    return np.vstack(rows)


def test_temporal_training_reuses_cache(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")

    data_path = tmp_path / "synth.csv"
    save_synthetic_dataset(
        data_path,
        n_authors=20,
        months=6,
        random_seed=5,
        difficulty="hard",
    )

    monkeypatch.setattr(
        train_temporal,
        "encode_texts_to_embeddings",
        _fake_encode_texts_to_embeddings,
    )

    config = TemporalTrainConfig(
        input_path=str(data_path),
        output_dir=str(tmp_path / "artifacts"),
        test_size=0.25,
        random_seed=5,
        max_features=5000,
        encoder_model="distilbert-base-uncased",
        max_length=64,
        batch_size=8,
        cache_embeddings=True,
        cache_dir=str(tmp_path / "cache"),
        gru_hidden_dim=16,
        gru_layers=1,
        dropout=0.1,
        lr=0.001,
        epochs=1,
    )

    first = train_temporal.run_training_temporal(config)
    second = train_temporal.run_training_temporal(config)

    assert bool(first["used_cache"]) is False
    assert bool(second["used_cache"]) is True
