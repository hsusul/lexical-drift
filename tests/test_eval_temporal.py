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
    assert str(first["model_type"]) == "gru"

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
    assert "model_type" in payload
    assert "plot_paths" in payload
    assert "per_month_csv_path" in payload
    assert "run_metadata_path" in payload
    assert "git_commit_hash" in payload
    assert "timestamp_iso" in payload
    assert "dataset_hash" in payload
    assert "config_hash" in payload
    plot_paths_payload = dict(payload["plot_paths"])
    assert Path(str(plot_paths_payload["per_month_metrics_path"])).exists()
    assert Path(str(plot_paths_payload["threshold_over_time_path"])).exists()
    assert Path(str(plot_paths_payload["pred_rate_over_time_path"])).exists()
    assert Path(str(plot_paths_payload["embedding_drift_over_time_path"])).exists()
    assert Path(str(plot_paths_payload["drift_vs_accuracy_delta_path"])).exists()
    assert Path(str(payload["per_month_csv_path"])).exists()
    run_metadata_path = Path(str(payload["run_metadata_path"]))
    assert run_metadata_path.exists()
    run_metadata_payload = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    assert isinstance(run_metadata_payload["seed"], int)
    assert isinstance(run_metadata_payload["encoder_model"], str)
    assert isinstance(run_metadata_payload["model_type"], str)
    assert isinstance(run_metadata_payload["config_hash"], str)

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
        assert "cosine_drift" in month_entry
        assert "l2_drift" in month_entry
        assert "variance_shift" in month_entry
        assert "accuracy_delta_from_ref" in month_entry
        assert "f1_delta_from_ref" in month_entry
        roc_auc = month_entry["roc_auc"]
        pr_auc = month_entry["pr_auc"]
        assert roc_auc is None or isinstance(roc_auc, float)
        assert pr_auc is None or isinstance(pr_auc, float)
        assert isinstance(month_entry["cosine_drift"], float)
        assert isinstance(month_entry["l2_drift"], float)
        assert isinstance(month_entry["variance_shift"], float)
        assert isinstance(month_entry["accuracy_delta_from_ref"], float)
        assert isinstance(month_entry["f1_delta_from_ref"], float)

    json.dumps(payload)
    json.dumps(first)
    assert Path(str(first["per_month_csv_path"])).exists()
    assert isinstance(first["final_month_probs"], list)
    assert isinstance(first["final_month_pred_counts"]["pred_0"], int)
    assert isinstance(first["final_month_confusion"]["tp"], int)
    assert isinstance(first["git_commit_hash"], str)
    assert isinstance(first["timestamp_iso"], str)
    assert isinstance(first["dataset_hash"], str)
    assert isinstance(first["config_hash"], str)
    assert isinstance(first["per_month_summary"]["accuracy_min"], float)
    assert isinstance(first["chosen_threshold"], float)
    assert isinstance(first["calibration_metric"], str)
    assert isinstance(first["final_month_threshold"], float)
    assert Path(str(first["run_metadata_path"])).exists()
    assert isinstance(first["per_month"][0]["pred_pos_rate"], float)
    assert isinstance(first["per_month"][0]["tn"], int)
    assert isinstance(first["per_month"][0]["threshold_used"], float)
    assert isinstance(first["per_month"][0]["precision"], float)
    assert isinstance(first["per_month"][0]["recall"], float)
    assert isinstance(first["per_month"][0]["specificity"], float)
    assert isinstance(first["per_month"][0]["balanced_accuracy"], float)
    assert "plot_paths" in first
    plot_paths = dict(first["plot_paths"])
    assert Path(str(plot_paths["per_month_metrics_path"])).exists()
    assert Path(str(plot_paths["threshold_over_time_path"])).exists()
    assert Path(str(plot_paths["pred_rate_over_time_path"])).exists()
    assert Path(str(plot_paths["embedding_drift_over_time_path"])).exists()
    assert Path(str(plot_paths["drift_vs_accuracy_delta_path"])).exists()
    assert first["per_month"][0]["roc_auc"] is None or isinstance(
        first["per_month"][0]["roc_auc"], float
    )
    assert first["per_month"][0]["pr_auc"] is None or isinstance(
        first["per_month"][0]["pr_auc"], float
    )
    assert isinstance(first["per_month"][0]["cosine_drift"], float)
    assert isinstance(first["per_month"][0]["l2_drift"], float)
    assert isinstance(first["per_month"][0]["variance_shift"], float)
    assert first["per_month"][0]["accuracy_delta_from_ref"] == pytest.approx(0.0)
    assert first["per_month"][0]["f1_delta_from_ref"] == pytest.approx(0.0)

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


def test_eval_temporal_baseline_lr_and_plots(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "synth_baseline.csv"
    save_synthetic_dataset(
        data_path,
        n_authors=24,
        months=6,
        random_seed=19,
        difficulty="hard",
    )

    monkeypatch.setattr(
        eval_temporal,
        "encode_texts_to_embeddings",
        _fake_encode_texts_to_embeddings,
    )

    config = EvalTemporalConfig(
        input_path=str(data_path),
        output_dir=str(tmp_path / "artifacts_baseline"),
        random_seed=19,
        model_type="baseline_lr",
        encoder_model="distilbert-base-uncased",
        max_length=64,
        batch_size=8,
        cache_embeddings=True,
        cache_dir=str(tmp_path / "cache_baseline"),
        train_months=3,
        gru_hidden_dim=16,
        gru_layers=1,
        dropout=0.1,
        lr=0.001,
        epochs=1,
        test_size=0.25,
    )

    result = eval_temporal.run_eval_temporal(config)
    assert str(result["model_type"]) == "baseline_lr"
    model_path = Path(str(result["model_path"]))
    assert model_path.suffix == ".joblib"
    assert model_path.exists()

    plot_paths = dict(result["plot_paths"])
    assert Path(str(plot_paths["per_month_metrics_path"])).exists()
    assert Path(str(plot_paths["threshold_over_time_path"])).exists()
    assert Path(str(plot_paths["pred_rate_over_time_path"])).exists()
    assert Path(str(plot_paths["embedding_drift_over_time_path"])).exists()
    assert Path(str(plot_paths["drift_vs_accuracy_delta_path"])).exists()


def test_eval_temporal_attention_model(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")

    data_path = tmp_path / "synth_attention.csv"
    save_synthetic_dataset(
        data_path,
        n_authors=24,
        months=6,
        random_seed=23,
        difficulty="hard",
    )
    monkeypatch.setattr(
        eval_temporal,
        "encode_texts_to_embeddings",
        _fake_encode_texts_to_embeddings,
    )

    config = EvalTemporalConfig(
        input_path=str(data_path),
        output_dir=str(tmp_path / "artifacts_attention"),
        random_seed=23,
        model_type="attention",
        encoder_model="distilbert-base-uncased",
        max_length=64,
        batch_size=8,
        cache_embeddings=True,
        cache_dir=str(tmp_path / "cache_attention"),
        train_months=3,
        gru_hidden_dim=16,
        gru_layers=1,
        dropout=0.1,
        lr=0.001,
        epochs=1,
        test_size=0.25,
    )

    result = eval_temporal.run_eval_temporal(config)
    assert str(result["model_type"]) == "attention"
    model_path = Path(str(result["model_path"]))
    assert model_path.exists()
    assert model_path.suffix == ".pt"
    assert isinstance(result["final_accuracy"], float)
    assert isinstance(result["final_f1"], float)
