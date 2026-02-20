from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lexical_drift.config import EvalTemporalConfig
from lexical_drift.eval import eval_temporal
from lexical_drift.eval.eval_temporal_sweep import aggregate_sweep_metrics, run_eval_temporal_sweep


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


def test_eval_temporal_sweep_runs_and_aggregates(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(
        eval_temporal,
        "encode_texts_to_embeddings",
        _fake_encode_texts_to_embeddings,
    )

    config_template = EvalTemporalConfig(
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

    results_path = tmp_path / "artifacts" / "eval_temporal_sweep.jsonl"
    sweep = run_eval_temporal_sweep(
        config_template=config_template,
        seeds=[1, 2],
        n_authors=20,
        months=6,
        difficulty="hard",
        artifact_root=tmp_path / "artifacts",
        results_path=results_path,
    )

    assert Path(str(sweep["results_path"])).exists()
    assert int(sweep["total_runs"]) == 2
    assert int(sweep["success_count"]) == 2
    assert int(sweep["failure_count"]) == 0

    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    for record in records:
        assert record["status"] == "ok"
        assert record["model_type"] == "gru"
        assert "seed" in record
        assert "final_month_index" in record
        assert "final_accuracy" in record
        assert "final_f1" in record
        assert "final_roc_auc" in record
        assert "final_pr_auc" in record
        assert "threshold_mode" in record
        assert "chosen_threshold" in record
        assert "cache_fingerprint" in record
        assert "used_cache" in record
        assert isinstance(record["per_month"], list)
        assert len(record["per_month"]) == 3
        for month_entry in record["per_month"]:
            assert "precision" in month_entry
            assert "recall" in month_entry
            assert "specificity" in month_entry
            assert "balanced_accuracy" in month_entry

    summary = aggregate_sweep_metrics(records)
    assert isinstance(summary, dict)
    assert int(summary["total_runs"]) == 2
    assert int(summary["success_count"]) == 2
    assert int(summary["failure_count"]) == 0
    assert isinstance(summary["final_month_summary"], dict)
    assert isinstance(summary["all_eval_months_summary"], dict)
    assert isinstance(summary["final_month_summary"]["accuracy"], dict)
    assert isinstance(summary["all_eval_months_summary"]["f1"], dict)
    assert isinstance(summary["final_month_summary"]["balanced_accuracy"], dict)

    run_dir = tmp_path / "artifacts" / "eval_sweep_runs" / "seed_1"
    assert (run_dir / "per_month_metrics.png").exists()
    assert (run_dir / "threshold_over_time.png").exists()
    assert (run_dir / "pred_rate_over_time.png").exists()
