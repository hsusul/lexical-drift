from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from lexical_drift.config import EvalTemporalConfig
from lexical_drift.eval import eval_temporal
from lexical_drift.eval.ablation_train_months import run_ablation_train_months


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


def test_ablation_train_months_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        eval_temporal,
        "encode_texts_to_embeddings",
        _fake_encode_texts_to_embeddings,
    )

    config = EvalTemporalConfig(
        input_path=str(tmp_path / "placeholder.csv"),
        output_dir=str(tmp_path / "placeholder_out"),
        random_seed=0,
        model_type="baseline_lr",
        encoder_model="distilbert-base-uncased",
        max_length=64,
        batch_size=8,
        cache_embeddings=True,
        cache_dir=str(tmp_path / "placeholder_cache"),
        train_months=3,
        gru_hidden_dim=16,
        gru_layers=1,
        dropout=0.1,
        lr=0.001,
        epochs=1,
        threshold_mode="fixed",
        fixed_threshold=0.5,
        calibration_metric="balanced_accuracy",
        test_size=0.2,
    )

    result = run_ablation_train_months(
        config_template=config,
        train_months_values=[2, 3],
        seeds=[1, 2],
        n_authors=30,
        months=6,
        difficulty="hard",
        artifact_root=tmp_path / "artifacts",
    )

    summary_path = Path(str(result["summary_path"]))
    plot_path = Path(str(result["plot_path"]))
    assert summary_path.exists()
    assert plot_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["train_months_values"] == [2, 3]
    assert payload["seeds"] == [1, 2]
    assert isinstance(payload["rows"], list)
    assert len(payload["rows"]) == 2
    for row in payload["rows"]:
        assert "train_months" in row
        assert "final_accuracy_mean" in row
        assert "final_f1_mean" in row
        assert "final_balanced_accuracy_mean" in row
        assert "cosine_drift_mean" in row
        assert "l2_drift_mean" in row
        assert "variance_shift_mean" in row
