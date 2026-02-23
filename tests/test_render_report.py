from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from lexical_drift.config import EvalTemporalConfig
from lexical_drift.eval import eval_temporal
from lexical_drift.eval.eval_temporal_compare import run_eval_temporal_compare
from lexical_drift.eval.report import render_compare_report


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


def test_render_report_from_compare_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        eval_temporal,
        "encode_texts_to_embeddings",
        _fake_encode_texts_to_embeddings,
    )

    base_config = EvalTemporalConfig(
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
    config_a = base_config
    config_b = replace(
        base_config,
        threshold_mode="calibrate_each_month",
        calibration_metric="youden_j",
    )

    compare_result = run_eval_temporal_compare(
        config_a_template=config_a,
        config_b_template=config_b,
        config_a_path=tmp_path / "a.yaml",
        config_b_path=tmp_path / "b.yaml",
        seeds=[1, 2],
        n_authors=30,
        months=6,
        difficulty="hard",
        artifact_root=tmp_path / "artifacts",
    )

    report_path = tmp_path / "docs" / "report.md"
    output_path = render_compare_report(
        compare_summary_path=compare_result["summary_path"],
        out_path=report_path,
    )
    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "# lexical-drift Evaluation Report" in text
    assert "## Configurations" in text
    assert "## Significance (Final Month)" in text
    assert "## Drift-Performance Correlation" in text
    assert "## Artifact Paths" in text
