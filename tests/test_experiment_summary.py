from __future__ import annotations

import json
from pathlib import Path

import yaml

from lexical_drift.eval.experiment_summary import run_summarize_experiments


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_summarize_experiments_writes_markdown_and_best_configs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    artifact_root = tmp_path / "artifacts" / "experiment_runs"
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    (config_dir / "train_e2e_temporal.yaml").write_text(
        yaml.safe_dump(
            {
                "input_path": "data/raw/synth.csv",
                "output_dir": "artifacts/e2e",
                "loss_type": "bce",
                "pos_weight": None,
                "focal_gamma": 2.0,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (config_dir / "eval_e2e_temporal_calib.yaml").write_text(
        yaml.safe_dump(
            {
                "input_path": "data/raw/synth.csv",
                "output_dir": "artifacts/e2e",
                "threshold_mode": "calibrate_on_val",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _write_json(
        artifact_root / "e2e_sweep_runs" / "e2e_sweep_summary.json",
        {
            "per_metric": {
                "f1": {"mean": 0.71, "std": 0.03, "min": 0.67, "max": 0.75},
                "pr_auc": {"mean": 0.78, "std": 0.02, "min": 0.74, "max": 0.81},
                "roc_auc": {"mean": 0.8, "std": 0.01, "min": 0.78, "max": 0.82},
                "balanced_accuracy": {
                    "mean": 0.7,
                    "std": 0.02,
                    "min": 0.66,
                    "max": 0.73,
                },
                "brier_score": {"mean": 0.19, "std": 0.01, "min": 0.18, "max": 0.21},
                "ece": {"mean": 0.06, "std": 0.01, "min": 0.04, "max": 0.07},
                "chosen_threshold": {
                    "mean": 0.42,
                    "std": 0.03,
                    "min": 0.38,
                    "max": 0.46,
                },
            }
        },
    )
    (artifact_root / "e2e_sweep_runs" / "e2e_sweep_records.csv").write_text(
        "seed,f1\n1,0.7\n2,0.72\n",
        encoding="utf-8",
    )
    _write_json(
        artifact_root / "e2e_sweep_runs" / "threshold_stability.json",
        {
            "chosen_threshold_variance": 0.0009,
            "threshold_f1_variance_correlation": 0.12,
        },
    )

    _write_json(
        artifact_root / "ablation_time_embeddings" / "ablation_summary.json",
        {
            "delta_f1_stats": {
                "mean": 0.023,
                "std": 0.01,
                "min": 0.01,
                "max": 0.04,
                "n": 3,
            },
            "paired_t_test": {"t_stat": 2.21, "p_value": 0.041, "n": 3},
            "bootstrap_ci_95": {"mean": 0.023, "low": 0.006, "high": 0.04, "n": 3},
            "artifact_paths": {
                "deltas_csv_path": (
                    "artifacts/experiment_runs/ablation_time_embeddings/ablation_deltas.csv"
                ),
                "ablation_delta_plot_path": (
                    "artifacts/experiment_runs/ablation_time_embeddings/ablation_delta_plot.png"
                ),
            },
        },
    )

    (artifact_root / "ablation_loss").mkdir(parents=True, exist_ok=True)
    (artifact_root / "ablation_loss" / "loss_grid_results.csv").write_text(
        "loss_type,f1_mean\nfocal,0.72\n",
        encoding="utf-8",
    )
    _write_json(
        artifact_root / "ablation_loss" / "summary.json",
        {
            "rows": [
                {
                    "loss_label": "bce",
                    "loss_type": "bce",
                    "pos_weight": None,
                    "focal_gamma": 2.0,
                    "f1_mean": 0.70,
                    "pr_auc_mean": 0.80,
                },
                {
                    "loss_label": "weighted_bce",
                    "loss_type": "weighted_bce",
                    "pos_weight": 2.0,
                    "focal_gamma": 2.0,
                    "f1_mean": 0.72,
                    "pr_auc_mean": 0.78,
                },
                {
                    "loss_label": "focal",
                    "loss_type": "focal",
                    "pos_weight": 2.0,
                    "focal_gamma": 4.0,
                    "f1_mean": 0.72,
                    "pr_auc_mean": 0.82,
                },
            ],
            "loss_grid_results_csv": (
                "artifacts/experiment_runs/ablation_loss/loss_grid_results.csv"
            ),
        },
    )

    result = run_summarize_experiments(
        artifact_root=artifact_root,
        config_dir=config_dir,
    )

    summary_path = Path(result["summary_markdown_path"])
    assert summary_path.exists()
    content = summary_path.read_text(encoding="utf-8")
    assert "## Baseline Sweep (E2E)" in content
    assert "## Time-Embedding Ablation" in content
    assert "## Loss Ablation Top Configurations" in content
    assert "## Generated Best Configs" in content
    assert "chosen_threshold" in content

    train_best_path = Path(result["train_best_config_path"])
    eval_best_path = Path(result["eval_best_config_path"])
    promoted_train_path = Path(result["promoted_train_best_config_path"])
    promoted_eval_path = Path(result["promoted_eval_best_config_path"])
    assert train_best_path.exists()
    assert eval_best_path.exists()
    assert promoted_train_path.exists()
    assert promoted_eval_path.exists()

    train_best_text = train_best_path.read_text(encoding="utf-8")
    assert "# Auto-generated by lexdrift summarize-experiments." in train_best_text

    train_best_payload = yaml.safe_load(train_best_text)
    assert train_best_payload["loss_type"] == "focal"
    assert train_best_payload["pos_weight"] == 2.0
    assert train_best_payload["focal_gamma"] == 4.0
