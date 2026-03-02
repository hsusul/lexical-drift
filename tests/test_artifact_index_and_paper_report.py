from __future__ import annotations

import json
from pathlib import Path

from lexical_drift.eval.artifact_index import run_index_artifacts
from lexical_drift.eval.paper_report import run_render_paper_report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_index_artifacts_writes_markdown(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts" / "experiment_runs"

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
        "seed,f1\n1,0.7\n",
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
                "mean": 0.02,
                "std": 0.01,
                "min": 0.01,
                "max": 0.03,
                "n": 2,
            },
            "paired_t_test": {"t_stat": 2.0, "p_value": 0.05, "n": 2},
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
    _write_json(
        artifact_root / "ablation_loss" / "summary.json",
        {
            "rows": [
                {
                    "loss_type": "focal",
                    "pos_weight": 2.0,
                    "focal_gamma": 4.0,
                    "f1_mean": 0.72,
                    "pr_auc_mean": 0.82,
                }
            ],
            "loss_grid_results_csv": (
                "artifacts/experiment_runs/ablation_loss/loss_grid_results.csv"
            ),
        },
    )
    (artifact_root / "EXPERIMENT_SUMMARY.md").write_text(
        "# lexical-drift Experiment Summary\n",
        encoding="utf-8",
    )

    result = run_index_artifacts(artifact_root=artifact_root)
    index_path = Path(result["index_path"])
    assert index_path.exists()
    text = index_path.read_text(encoding="utf-8")
    assert "# lexical-drift Artifact Index" in text
    assert "## Latest Paths" in text
    assert "## Top Metrics" in text
    assert "## Time Ablation" in text
    assert "## Loss Best Config" in text


def test_render_paper_report_writes_markdown_with_figures(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts" / "experiment_runs"
    artifact_root.mkdir(parents=True, exist_ok=True)

    (artifact_root / "EXPERIMENT_SUMMARY.md").write_text(
        "\n".join(
            [
                "# lexical-drift Experiment Summary",
                "",
                "## Baseline Sweep (E2E)",
                "",
                "| metric | mean±std |",
                "|---|---:|",
                "| f1 | 0.71±0.03 |",
                "",
                "## Time-Embedding Ablation",
                "",
                "| metric | value |",
                "|---|---:|",
                "| delta_f1 mean±std | 0.0200±0.0100 |",
                "",
                "## Loss Ablation Top Configurations",
                "",
                "| rank | loss_type | pos_weight | focal_gamma | f1_mean | pr_auc_mean |",
                "|---:|---|---:|---:|---:|---:|",
                "| 1 | focal | 2.0 | 4.0 | 0.7200 | 0.8200 |",
            ]
        ),
        encoding="utf-8",
    )
    (artifact_root / "INDEX.md").write_text("# index\n", encoding="utf-8")
    plot_path = artifact_root / "ablation_time_embeddings" / "ablation_delta_plot.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    out_path = tmp_path / "docs" / "REPORT.md"
    result = run_render_paper_report(artifact_root=artifact_root, out_path=out_path)

    assert Path(result["report_path"]).exists()
    text = out_path.read_text(encoding="utf-8")
    assert "# lexical-drift Report" in text
    assert "## Headline Metrics" in text
    assert "## Ablation Highlights" in text
    assert "## Figures" in text
    assert f"![time-ablation]({plot_path})" in text
