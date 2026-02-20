from __future__ import annotations

import hashlib
import json
from dataclasses import replace
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
        vec = np.frombuffer(digest, dtype=np.uint8)[:32].astype(np.float32) / 255.0
        rows.append(vec)
    return np.vstack(rows)


def _fake_encode_texts_with_month_signal(
    texts: list[str],
    model_name: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    _ = (model_name, max_length, batch_size)
    rows: list[np.ndarray] = []
    for index, text in enumerate(texts):
        digest = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(digest, dtype=np.uint8)[:32].astype(np.float32) / 255.0
        month_signal = np.float32((index % 12) / 11.0)
        vec[0] = month_signal
        rows.append(vec)
    return np.vstack(rows)


def test_eval_temporal_per_month_metrics_and_cache(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")

    data_path = tmp_path / "synth.csv"
    save_synthetic_dataset(
        data_path,
        n_authors=20,
        months=6,
        random_seed=11,
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
        random_seed=11,
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
        test_size=0.25,
    )

    first = eval_temporal.run_eval_temporal(config)
    second = eval_temporal.run_eval_temporal(config)

    per_month = first["per_month"]
    assert isinstance(per_month, list)
    assert len(per_month) == 3
    assert [int(entry["month_index"]) for entry in per_month] == [3, 4, 5]
    assert bool(first["used_cache"]) is False
    assert bool(second["used_cache"]) is True

    metrics_path = Path(str(first["metrics_path"]))
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "per_month" in payload
    assert "final_month" in payload
    assert "summary" in payload
    assert "per_month_summary" in payload
    assert "threshold_mode" in payload
    assert "fixed_threshold" in payload
    assert "chosen_threshold" in payload
    assert "calibration_metric" in payload
    assert "final_month_threshold" in payload
    assert "final_month_probs" in payload
    assert "final_month_pred_counts" in payload
    assert "final_month_pred_rates" in payload
    assert "final_month_confusion" in payload
    assert "cache_path" in payload
    assert "used_cache" in payload

    for month_entry in payload["per_month"]:
        assert "month_index" in month_entry
        assert "accuracy" in month_entry
        assert "f1" in month_entry
        assert "precision" in month_entry
        assert "recall" in month_entry
        assert "specificity" in month_entry
        assert "balanced_accuracy" in month_entry
        assert "roc_auc" in month_entry
        assert "pr_auc" in month_entry
        assert "true_pos_rate" in month_entry
        assert "pred_pos_rate" in month_entry
        assert "mean_pred_prob" in month_entry
        assert "threshold_used" in month_entry
        assert "tn" in month_entry
        assert "fp" in month_entry
        assert "fn" in month_entry
        assert "tp" in month_entry
        roc_auc = month_entry["roc_auc"]
        pr_auc = month_entry["pr_auc"]
        assert roc_auc is None or isinstance(roc_auc, float)
        assert pr_auc is None or isinstance(pr_auc, float)

    json.dumps(payload)
    json.dumps(first)
    assert isinstance(first["final_month_probs"], list)
    assert isinstance(first["final_month_pred_counts"]["pred_0"], int)
    assert isinstance(first["final_month_confusion"]["tp"], int)
    assert isinstance(first["per_month_summary"]["accuracy_min"], float)
    assert isinstance(first["chosen_threshold"], float)
    assert isinstance(first["calibration_metric"], str)
    assert isinstance(first["final_month_threshold"], float)
    assert isinstance(first["per_month"][0]["pred_pos_rate"], float)
    assert isinstance(first["per_month"][0]["tn"], int)
    assert isinstance(first["per_month"][0]["threshold_used"], float)
    assert isinstance(first["per_month"][0]["precision"], float)
    assert isinstance(first["per_month"][0]["recall"], float)
    assert isinstance(first["per_month"][0]["specificity"], float)
    assert isinstance(first["per_month"][0]["balanced_accuracy"], float)
    assert first["per_month"][0]["roc_auc"] is None or isinstance(
        first["per_month"][0]["roc_auc"], float
    )
    assert first["per_month"][0]["pr_auc"] is None or isinstance(
        first["per_month"][0]["pr_auc"], float
    )

    changed = eval_temporal.run_eval_temporal(replace(config, max_length=96))
    assert bool(changed["used_cache"]) is False
    assert str(changed["cache_fingerprint"]) != str(second["cache_fingerprint"])


def test_eval_temporal_calibrate_each_month_thresholds(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")

    data_path = tmp_path / "synth.csv"
    save_synthetic_dataset(
        data_path,
        n_authors=36,
        months=10,
        random_seed=17,
        difficulty="hard",
    )

    monkeypatch.setattr(
        eval_temporal,
        "encode_texts_to_embeddings",
        _fake_encode_texts_with_month_signal,
    )

    config = EvalTemporalConfig(
        input_path=str(data_path),
        output_dir=str(tmp_path / "artifacts_each"),
        random_seed=17,
        encoder_model="distilbert-base-uncased",
        max_length=64,
        batch_size=8,
        cache_embeddings=False,
        cache_dir=str(tmp_path / "cache_each"),
        train_months=4,
        gru_hidden_dim=16,
        gru_layers=1,
        dropout=0.1,
        lr=0.001,
        epochs=1,
        threshold_mode="calibrate_each_month",
        calibration_metric="balanced_accuracy",
        test_size=0.25,
    )

    result = eval_temporal.run_eval_temporal(config)
    thresholds = [float(entry["threshold_used"]) for entry in result["per_month"]]
    assert thresholds
    assert all(0.0 <= value <= 1.0 for value in thresholds)
    assert len({round(value, 2) for value in thresholds}) > 1

    metrics_path = Path(str(result["metrics_path"]))
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "final_month_threshold" in payload
    assert float(payload["final_month_threshold"]) == pytest.approx(thresholds[-1])
