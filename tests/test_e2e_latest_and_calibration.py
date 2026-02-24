from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from lexical_drift.config import EvalE2EConfig, TrainE2EConfig
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.eval.eval_e2e_sweep import run_eval_e2e_sweep
from lexical_drift.train import e2e_temporal
from tests._requires_torch import requires_torch


def _hash_to_vector(text: str, dim: int = 32) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    data = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
    if data.size >= dim:
        return data[:dim]
    repeat = int(np.ceil(dim / data.size))
    return np.tile(data, repeat)[:dim]


def _build_fake_temporal_encoder(torch):
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
                seq_vectors = np.vstack(
                    [_hash_to_vector(text, dim=self._output_dim) for text in sequence]
                )
                rows.append(seq_vectors)
            matrix = np.asarray(rows, dtype=np.float32)
            tensor = torch.from_numpy(matrix).to(device=device, dtype=torch.float32)
            return tensor * self.scale

    return FakeTemporalEncoder


def _build_train_config(data_path: Path, output_dir: Path) -> TrainE2EConfig:
    return TrainE2EConfig(
        input_path=str(data_path),
        output_dir=str(output_dir),
        random_seed=7,
        encoder_model="fake-encoder",
        max_length=32,
        batch_size=4,
        train_months=3,
        gru_hidden_dim=16,
        gru_layers=1,
        dropout=0.1,
        lr=0.001,
        epochs=1,
        test_size=0.25,
        pooling="cls",
        freeze_encoder=False,
        pretrained_encoder_path="",
        use_time_embeddings=True,
        loss_type="bce",
        pos_weight=None,
        focal_gamma=2.0,
    )


@requires_torch
def test_train_e2e_writes_latest_pointer(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        e2e_temporal,
        "TemporalEncoder",
        _build_fake_temporal_encoder(torch),
    )

    data_path = tmp_path / "synth.csv"
    save_synthetic_dataset(
        out_path=data_path,
        n_authors=20,
        months=6,
        random_seed=7,
        difficulty="hard",
    )

    train_config = _build_train_config(data_path, tmp_path / "artifacts" / "e2e")
    result = e2e_temporal.run_train_e2e(train_config)

    pointer_path = Path(str(result["latest_pointer_path"]))
    assert pointer_path.exists()
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert isinstance(payload["config_hash"], str)
    assert isinstance(payload["timestamp_iso"], str)
    model_path_value = Path(str(payload["model_path"]))
    if not model_path_value.is_absolute():
        model_path_value = (Path.cwd() / model_path_value).resolve()
    assert model_path_value.exists()


@requires_torch
def test_eval_e2e_use_latest_cli(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from lexical_drift import cli as cli_module

    monkeypatch.setattr(
        e2e_temporal,
        "TemporalEncoder",
        _build_fake_temporal_encoder(torch),
    )
    monkeypatch.setattr(
        cli_module,
        "_dependency_available",
        lambda module_name: module_name in {"torch", "transformers"},
    )

    data_path = tmp_path / "synth.csv"
    save_synthetic_dataset(
        out_path=data_path,
        n_authors=20,
        months=6,
        random_seed=7,
        difficulty="hard",
    )

    train_output_dir = tmp_path / "artifacts" / "e2e"
    train_config_path = tmp_path / "train_e2e.yaml"
    train_config_path.write_text(
        yaml.safe_dump(
            {
                "input_path": str(data_path),
                "output_dir": str(train_output_dir),
                "random_seed": 7,
                "encoder_model": "fake-encoder",
                "max_length": 32,
                "batch_size": 4,
                "train_months": 3,
                "gru_hidden_dim": 16,
                "gru_layers": 1,
                "dropout": 0.1,
                "lr": 0.001,
                "epochs": 1,
                "test_size": 0.25,
                "pooling": "cls",
                "freeze_encoder": False,
                "pretrained_encoder_path": "",
                "use_time_embeddings": True,
                "loss_type": "bce",
                "pos_weight": None,
                "focal_gamma": 2.0,
            }
        ),
        encoding="utf-8",
    )
    eval_config_path = tmp_path / "eval_e2e.yaml"
    eval_config_path.write_text(
        yaml.safe_dump(
            {
                "input_path": str(data_path),
                "output_dir": str(train_output_dir),
                "random_seed": 7,
                "encoder_model": "fake-encoder",
                "max_length": 32,
                "batch_size": 4,
                "train_months": 3,
                "checkpoint_path": "",
                "test_size": 0.25,
                "pooling": "cls",
                "threshold_mode": "fixed",
                "fixed_threshold": 0.5,
                "calibration_metric": "balanced_accuracy",
                "pretrained_encoder_path": "",
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    train_cmd = runner.invoke(cli_module.app, ["train-e2e", "--config", str(train_config_path)])
    assert train_cmd.exit_code == 0

    eval_cmd = runner.invoke(
        cli_module.app,
        ["eval-e2e", "--config", str(eval_config_path), "--use-latest"],
    )
    assert eval_cmd.exit_code == 0
    assert "using latest checkpoint" in eval_cmd.output
    assert "threshold mode=fixed" in eval_cmd.output


@requires_torch
def test_eval_e2e_threshold_calibration_improves_f1(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        e2e_temporal,
        "TemporalEncoder",
        _build_fake_temporal_encoder(torch),
    )

    data_path = tmp_path / "synth.csv"
    save_synthetic_dataset(
        out_path=data_path,
        n_authors=30,
        months=6,
        random_seed=9,
        difficulty="hard",
    )

    train_config = _build_train_config(data_path, tmp_path / "artifacts" / "e2e")
    train_result = e2e_temporal.run_train_e2e(train_config)

    def _fake_predict_probs_for_month(
        *,
        encoder,
        head,
        device,
        author_ids,
        sequences_texts,
        labels,
        sequences_months,
        eval_indices,
        month_index,
        batch_size,
        time_embedding,
    ) -> np.ndarray:
        _ = (
            encoder,
            head,
            device,
            author_ids,
            sequences_texts,
            sequences_months,
            month_index,
            batch_size,
            time_embedding,
        )
        y_values = labels[eval_indices].astype(np.float32)
        base = np.where(y_values == 1.0, 0.43, 0.29).astype(np.float32)
        offsets = (eval_indices.astype(np.float32) % 3.0) * 0.01
        return np.clip(base + offsets, 0.01, 0.99)

    monkeypatch.setattr(e2e_temporal, "_predict_probs_for_month", _fake_predict_probs_for_month)

    fixed_config = EvalE2EConfig(
        input_path=str(data_path),
        output_dir=str(tmp_path / "artifacts" / "e2e"),
        random_seed=9,
        encoder_model="fake-encoder",
        max_length=32,
        batch_size=4,
        train_months=3,
        checkpoint_path=str(train_result["model_path"]),
        test_size=0.25,
        pooling="cls",
        threshold=0.5,
        threshold_mode="fixed",
        calibration_metric="f1",
        fixed_threshold=0.5,
        threshold_min=0.2,
        threshold_max=0.8,
        n_thresholds=13,
        pretrained_encoder_path="",
    )
    calibrated_config = EvalE2EConfig(
        input_path=str(data_path),
        output_dir=str(tmp_path / "artifacts" / "e2e"),
        random_seed=9,
        encoder_model="fake-encoder",
        max_length=32,
        batch_size=4,
        train_months=3,
        checkpoint_path=str(train_result["model_path"]),
        test_size=0.25,
        pooling="cls",
        threshold=0.5,
        threshold_mode="calibrate_on_val",
        calibration_metric="f1",
        fixed_threshold=0.5,
        threshold_min=0.2,
        threshold_max=0.8,
        n_thresholds=13,
        pretrained_encoder_path="",
    )

    fixed_result = e2e_temporal.run_eval_e2e(fixed_config)
    calibrated_result = e2e_temporal.run_eval_e2e(calibrated_config)

    assert fixed_result["chosen_threshold"] == pytest.approx(0.5)
    assert 0.2 <= calibrated_result["chosen_threshold"] <= 0.8
    assert calibrated_result["chosen_threshold"] < 0.5
    assert calibrated_result["final_f1"] > fixed_result["final_f1"]
    assert calibrated_result["threshold_mode"] == "calibrate_on_val"
    assert calibrated_result["per_month"][0]["threshold_used"] == pytest.approx(
        calibrated_result["chosen_threshold"]
    )
    metrics_path = Path(str(calibrated_result["metrics_path"]))
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    artifact_paths = dict(metrics_payload["artifact_paths"])
    curve_path = Path(str(artifact_paths["val_threshold_curve_json_path"]))
    curve_plot_path = Path(str(artifact_paths["val_threshold_curve_plot_path"]))
    assert curve_path.exists()
    assert curve_plot_path.exists()
    assert Path(str(artifact_paths["val_prob_hist_path"])).exists()
    assert Path(str(artifact_paths["test_prob_hist_path"])).exists()
    assert Path(str(artifact_paths["val_prob_summary_path"])).exists()
    assert Path(str(artifact_paths["test_prob_summary_path"])).exists()
    curve_payload = json.loads(curve_path.read_text(encoding="utf-8"))
    assert set(curve_payload) >= {
        "thresholds",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "youden_j",
        "chosen_threshold",
        "chosen_metric",
        "chosen_metric_value",
    }
    size = len(curve_payload["thresholds"])
    assert size == 13
    for key in ("f1", "precision", "recall", "balanced_accuracy", "youden_j"):
        assert len(curve_payload[key]) == size


@requires_torch
def test_eval_e2e_sweep_records_threshold_used(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        e2e_temporal,
        "TemporalEncoder",
        _build_fake_temporal_encoder(torch),
    )

    train_template = TrainE2EConfig(
        input_path=str(tmp_path / "placeholder.csv"),
        output_dir=str(tmp_path / "artifacts" / "e2e"),
        random_seed=3,
        encoder_model="fake-encoder",
        max_length=32,
        batch_size=4,
        train_months=3,
        gru_hidden_dim=16,
        gru_layers=1,
        dropout=0.1,
        lr=0.001,
        epochs=1,
        test_size=0.25,
        pooling="cls",
        freeze_encoder=False,
        pretrained_encoder_path="",
        use_time_embeddings=True,
        loss_type="bce",
        pos_weight=None,
        focal_gamma=2.0,
        train_eval_threshold_mode="fixed",
        train_eval_calibration_metric="f1",
    )
    eval_template = EvalE2EConfig(
        input_path=str(tmp_path / "placeholder.csv"),
        output_dir=str(tmp_path / "artifacts" / "e2e"),
        random_seed=3,
        encoder_model="fake-encoder",
        max_length=32,
        batch_size=4,
        train_months=3,
        test_size=0.25,
        checkpoint_path="",
        pooling="cls",
        threshold=0.5,
        threshold_mode="fixed",
        calibration_metric="balanced_accuracy",
        fixed_threshold=0.5,
        threshold_min=0.05,
        threshold_max=0.95,
        n_thresholds=101,
        pretrained_encoder_path="",
    )

    sweep_result = run_eval_e2e_sweep(
        train_config_template=train_template,
        eval_config_template=eval_template,
        seeds=[1, 2],
        n_authors=20,
        months=6,
        difficulty="hard",
        artifact_root=tmp_path / "artifacts",
        results_path=tmp_path / "artifacts" / "eval_e2e_sweep.jsonl",
    )
    assert sweep_result["success_count"] == 2
    records_path = Path(str(sweep_result["results_path"]))
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    for record in records:
        assert record["status"] == "ok"
        assert isinstance(record["final_month_threshold_used"], float)
        assert isinstance(record["chosen_threshold"], float)
        final_month_metrics = dict(record["final_month_metrics"])
        assert isinstance(final_month_metrics["threshold_used"], float)
