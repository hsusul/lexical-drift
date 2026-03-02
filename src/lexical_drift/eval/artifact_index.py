from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _latest_file(root: Path, pattern: str) -> Path | None:
    matches = [path for path in root.glob(pattern) if path.is_file()]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime_ns)
    return matches[-1]


def _format_metric(stats: object) -> str:
    if not isinstance(stats, dict):
        return "na"
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None:
        return "na"
    return f"{float(mean):.4f}±{float(std):.4f}"


def _best_loss_row(loss_summary: dict[str, Any]) -> dict[str, Any] | None:
    best = loss_summary.get("best_configuration")
    if isinstance(best, dict):
        return dict(best)

    rows_raw = loss_summary.get("rows")
    rows = (
        [dict(row) for row in rows_raw if isinstance(row, dict)]
        if isinstance(rows_raw, list)
        else []
    )
    if not rows:
        return None

    def sort_key(row: dict[str, Any]) -> tuple[float, float]:
        f1 = row.get("f1_mean")
        pr_auc = row.get("pr_auc_mean")
        f1_value = float(f1) if f1 is not None else float("-inf")
        pr_auc_value = float(pr_auc) if pr_auc is not None else float("-inf")
        return (f1_value, pr_auc_value)

    rows.sort(key=sort_key, reverse=True)
    return rows[0]


def run_index_artifacts(
    *,
    artifact_root: str | Path = "artifacts/experiment_runs",
) -> dict[str, str]:
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)

    sweep_summary_path = _latest_file(root, "**/e2e_sweep_summary.json")
    sweep_records_csv = _latest_file(root, "**/e2e_sweep_records.csv")
    threshold_stability_path = _latest_file(root, "**/threshold_stability.json")
    time_ablation_path = _latest_file(root, "**/ablation_time_embeddings/ablation_summary.json")
    loss_ablation_path = _latest_file(root, "**/ablation_loss/summary.json")
    experiment_summary_path = _latest_file(root, "**/EXPERIMENT_SUMMARY.md")

    sweep_summary = _load_json(sweep_summary_path) if sweep_summary_path is not None else {}
    time_summary = _load_json(time_ablation_path) if time_ablation_path is not None else {}
    loss_summary = _load_json(loss_ablation_path) if loss_ablation_path is not None else {}

    lines: list[str] = [
        "# lexical-drift Artifact Index",
        "",
        f"- artifact_root: `{root}`",
        "",
        "## Latest Paths",
        "",
        f"- Sweep summary: `{sweep_summary_path if sweep_summary_path else 'na'}`",
        f"- Sweep records CSV: `{sweep_records_csv if sweep_records_csv else 'na'}`",
        (
            "- Threshold stability: "
            f"`{threshold_stability_path if threshold_stability_path else 'na'}`"
        ),
        f"- Time ablation summary: `{time_ablation_path if time_ablation_path else 'na'}`",
        f"- Loss ablation summary: `{loss_ablation_path if loss_ablation_path else 'na'}`",
        f"- Experiment summary: `{experiment_summary_path if experiment_summary_path else 'na'}`",
    ]

    per_metric = sweep_summary.get("per_metric") if isinstance(sweep_summary, dict) else None
    lines.extend(["", "## Top Metrics", ""])
    if isinstance(per_metric, dict):
        lines.extend(["| metric | mean±std |", "|---|---:|"])
        for metric in (
            "f1",
            "pr_auc",
            "roc_auc",
            "balanced_accuracy",
            "brier_score",
            "ece",
            "chosen_threshold",
        ):
            lines.append(f"| {metric} | {_format_metric(per_metric.get(metric))} |")
    else:
        lines.append("No sweep metrics available.")

    lines.extend(["", "## Time Ablation", ""])
    if isinstance(time_summary, dict) and time_summary:
        delta_stats = time_summary.get("delta_f1_stats")
        if isinstance(delta_stats, dict):
            lines.append(f"- delta_f1 mean±std: {_format_metric(delta_stats)}")
        t_test = time_summary.get("paired_t_test")
        if isinstance(t_test, dict):
            lines.append(
                "- paired t-test: "
                f"t={t_test.get('t_stat', 'na')} "
                f"p={t_test.get('p_value', 'na')} "
                f"n={t_test.get('n', 'na')}"
            )
        artifact_paths = time_summary.get("artifact_paths")
        if isinstance(artifact_paths, dict):
            lines.append(f"- delta CSV: `{artifact_paths.get('deltas_csv_path', 'na')}`")
            lines.append(f"- delta plot: `{artifact_paths.get('ablation_delta_plot_path', 'na')}`")
    else:
        lines.append("No time-ablation summary available.")

    lines.extend(["", "## Loss Best Config", ""])
    if isinstance(loss_summary, dict) and loss_summary:
        best = _best_loss_row(loss_summary)
        if best is not None:
            lines.append(
                "- best: "
                f"loss_type={best.get('loss_type', best.get('loss_label', 'na'))} "
                f"pos_weight={best.get('pos_weight', 'na')} "
                f"focal_gamma={best.get('focal_gamma', 'na')} "
                f"f1_mean={best.get('f1_mean', 'na')} "
                f"pr_auc_mean={best.get('pr_auc_mean', 'na')}"
            )
        csv_path = loss_summary.get("loss_grid_results_csv")
        if csv_path:
            lines.append(f"- grid CSV: `{csv_path}`")
    else:
        lines.append("No loss-ablation summary available.")

    index_path = root / "INDEX.md"
    index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    return {
        "index_path": str(index_path),
        "artifact_root": str(root),
    }
