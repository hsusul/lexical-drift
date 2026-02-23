from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lexical_drift.config import EvalTemporalConfig
from lexical_drift.eval import eval_temporal
from lexical_drift.eval.eval_temporal_compare import run_eval_temporal_compare


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


def _counting_fake_encoder_factory(counter: dict[str, int]):
    def _encoder(
        texts: list[str],
        model_name: str,
        max_length: int,
        batch_size: int,
    ) -> np.ndarray:
        counter["calls"] += 1
        return _fake_encode_texts_to_embeddings(
            texts=texts,
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
        )

    return _encoder


def test_eval_temporal_compare_summary_and_deltas(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(
        eval_temporal,
        "encode_texts_to_embeddings",
        _fake_encode_texts_to_embeddings,
    )

    base_config = EvalTemporalConfig(
        input_path=str(tmp_path / "placeholder.csv"),
        output_dir=str(tmp_path / "placeholder_out"),
        random_seed=0,
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

    config_a_path = tmp_path / "eval_a.yaml"
    config_b_path = tmp_path / "eval_b.yaml"
    config_a_path.write_text("placeholder: true\n", encoding="utf-8")
    config_b_path.write_text("placeholder: true\n", encoding="utf-8")

    result = run_eval_temporal_compare(
        config_a_template=config_a,
        config_b_template=config_b,
        config_a_path=config_a_path,
        config_b_path=config_b_path,
        seeds=[1, 2],
        n_authors=40,
        months=6,
        difficulty="hard",
        artifact_root=tmp_path / "artifacts",
    )

    summary_path = Path(str(result["summary_path"]))
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    required_keys = {
        "config_a_path",
        "config_b_path",
        "seeds",
        "n_authors",
        "months",
        "difficulty",
        "final_month_summary_a",
        "final_month_summary_b",
        "final_month_delta",
        "all_months_summary_a",
        "all_months_summary_b",
        "all_months_delta",
    }
    assert required_keys.issubset(payload)
    assert payload["summary_a"]["model_type"] == "gru"
    assert payload["summary_b"]["model_type"] == "gru"

    for section in ("final_month_delta", "all_months_delta"):
        values = payload[section]
        assert isinstance(values, dict)
        for metric in (
            "accuracy",
            "f1",
            "precision",
            "recall",
            "specificity",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
            "pred_pos_rate",
            "true_pos_rate",
            "threshold_used",
            "cosine_drift",
            "l2_drift",
            "variance_shift",
        ):
            assert metric in values
            metric_value = values[metric]
            assert metric_value is None or isinstance(metric_value, float)

    plot_dir_a = tmp_path / "artifacts" / "compare_runs" / "A" / "seed_1"
    plot_dir_b = tmp_path / "artifacts" / "compare_runs" / "B" / "seed_1"
    assert (plot_dir_a / "per_month_metrics.png").exists()
    assert (plot_dir_a / "threshold_over_time.png").exists()
    assert (plot_dir_a / "pred_rate_over_time.png").exists()
    assert (plot_dir_a / "embedding_drift_over_time.png").exists()
    assert (plot_dir_b / "per_month_metrics.png").exists()
    assert (plot_dir_b / "threshold_over_time.png").exists()
    assert (plot_dir_b / "pred_rate_over_time.png").exists()
    assert (plot_dir_b / "embedding_drift_over_time.png").exists()


def test_eval_temporal_compare_reuses_shared_cache(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    call_counter = {"calls": 0}
    monkeypatch.setattr(
        eval_temporal,
        "encode_texts_to_embeddings",
        _counting_fake_encoder_factory(call_counter),
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
        train_months=8,
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

    run_eval_temporal_compare(
        config_a_template=config_a,
        config_b_template=config_b,
        config_a_path=tmp_path / "eval_a.yaml",
        config_b_path=tmp_path / "eval_b.yaml",
        seeds=[1, 2],
        n_authors=40,
        months=12,
        difficulty="hard",
        artifact_root=tmp_path / "artifacts",
    )

    assert call_counter["calls"] == 2
