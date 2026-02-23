from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from lexical_drift.config import EvalTemporalConfig
from lexical_drift.datasets.real import load_real_dataset
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


def test_load_real_dataset_and_eval(tmp_path, monkeypatch) -> None:
    raw_path = tmp_path / "real_source.csv"
    frame = pd.DataFrame(
        {
            "author_id": ["a0"] * 6 + ["a1"] * 6 + ["b0"] * 6 + ["b1"] * 6,
            "month": [f"2024-{month:02d}-01" for month in range(1, 7)] * 4,
            "text": [
                f"author={author} month={month} sample text"
                for author in ("a0", "a1", "b0", "b1")
                for month in range(6)
            ],
            "label": [0] * 12 + [1] * 12,
        }
    )
    frame.to_csv(raw_path, index=False)

    normalized = load_real_dataset(name="sample_local", path=raw_path)
    assert list(normalized.columns) == ["author_id", "month_index", "text", "drift_label"]
    assert normalized["month_index"].min() == 0
    assert normalized["month_index"].max() == 5

    normalized_path = tmp_path / "real_normalized.csv"
    normalized.to_csv(normalized_path, index=False)

    monkeypatch.setattr(
        eval_temporal,
        "encode_texts_to_embeddings",
        _fake_encode_texts_to_embeddings,
    )
    config = EvalTemporalConfig(
        input_path=str(normalized_path),
        output_dir=str(tmp_path / "artifacts"),
        random_seed=0,
        model_type="baseline_lr",
        encoder_model="distilbert-base-uncased",
        max_length=64,
        batch_size=8,
        cache_embeddings=True,
        cache_dir=str(tmp_path / "cache"),
        train_months=3,
        gru_hidden_dim=16,
        gru_layers=1,
        dropout=0.1,
        lr=0.001,
        epochs=1,
        threshold_mode="fixed",
        fixed_threshold=0.5,
        calibration_metric="balanced_accuracy",
        test_size=0.5,
    )
    result = eval_temporal.run_eval_temporal(config)
    assert Path(str(result["metrics_path"])).exists()
