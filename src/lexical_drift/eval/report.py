from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from lexical_drift.eval.eval_temporal_sweep import aggregate_sweep_metrics


def _format_value(value: object) -> str:
    if value is None:
        return "na"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            records.append(json.loads(text))
    return records


def _build_metric_table(
    *,
    metrics_a: dict[str, object],
    metrics_b: dict[str, object],
) -> str:
    rows = ["| metric | A mean | B mean |", "|---|---:|---:|"]
    for metric in sorted(set(metrics_a) | set(metrics_b)):
        value_a = metrics_a.get(metric)
        value_b = metrics_b.get(metric)
        mean_a = value_a.get("mean") if isinstance(value_a, dict) else None
        mean_b = value_b.get("mean") if isinstance(value_b, dict) else None
        rows.append(f"| {metric} | {_format_value(mean_a)} | {_format_value(mean_b)} |")
    return "\n".join(rows)


def _build_single_summary_table(
    *,
    title: str,
    summary: dict[str, object],
) -> list[str]:
    lines = [f"### {title}", "", "| metric | mean | std | min | max |", "|---|---:|---:|---:|---:|"]
    for metric in sorted(summary):
        stats = summary.get(metric)
        if not isinstance(stats, dict):
            continue
        lines.append(
            "| "
            f"{metric} | "
            f"{_format_value(stats.get('mean'))} | "
            f"{_format_value(stats.get('std'))} | "
            f"{_format_value(stats.get('min'))} | "
            f"{_format_value(stats.get('max'))} |"
        )
    return lines


def _build_significance_table(stats: dict[str, object]) -> str:
    rows = [
        "| metric | mean_delta | ci_low | ci_high | p_value | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, value in stats.items():
        if not isinstance(value, dict):
            rows.append(f"| {metric} | na | na | na | na | 0 |")
            continue
        rows.append(
            "| "
            f"{metric} | "
            f"{_format_value(value.get('mean_delta'))} | "
            f"{_format_value(value.get('ci_low'))} | "
            f"{_format_value(value.get('ci_high'))} | "
            f"{_format_value(value.get('p_value'))} | "
            f"{_format_value(value.get('n'))} |"
        )
    return "\n".join(rows)


def _build_correlation_table(summary: dict[str, object]) -> str:
    rows = ["| pair | mean | std | min | max | n |", "|---|---:|---:|---:|---:|---:|"]
    for pair_key, value in summary.items():
        if not isinstance(value, dict):
            rows.append(f"| {pair_key} | na | na | na | na | 0 |")
            continue
        rows.append(
            "| "
            f"{pair_key} | "
            f"{_format_value(value.get('mean'))} | "
            f"{_format_value(value.get('std'))} | "
            f"{_format_value(value.get('min'))} | "
            f"{_format_value(value.get('max'))} | "
            f"{_format_value(value.get('n'))} |"
        )
    return "\n".join(rows)


def _find_optional_compare(artifact_root: Path) -> Path | None:
    candidates = [
        artifact_root / "e2e_compare_summary.json",
        artifact_root / "eval_e2e_compare_summary.json",
        artifact_root / "e2e_vs_frozen_compare_summary.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _latest_match(pattern_root: Path, pattern: str) -> Path | None:
    matches = sorted(pattern_root.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def render_compare_report(
    *,
    compare_summary_path: str | Path,
    out_path: str | Path,
) -> Path:
    summary_path = Path(compare_summary_path)
    payload = _load_json(summary_path)
    artifact_root = summary_path.parent

    summary_a = payload.get("summary_a", {})
    summary_b = payload.get("summary_b", {})
    final_month_summary_a = payload.get("final_month_summary_a", {})
    final_month_summary_b = payload.get("final_month_summary_b", {})
    final_month_stats = payload.get("final_month_stats", {})
    all_months_stats = payload.get("all_months_stats", {})
    corr_a = payload.get("drift_performance_correlation_a", {})
    corr_b = payload.get("drift_performance_correlation_b", {})
    run_meta_a = payload.get("per_run_metadata_a", [])
    run_meta_b = payload.get("per_run_metadata_b", [])

    lines = [
        "# lexical-drift Evaluation Report",
        "",
        "## Overview",
        "",
        f"- Generated at: {datetime.now(UTC).isoformat()}",
        f"- Primary compare summary: `{summary_path}`",
        "",
        "## Configurations",
        "",
        f"- Config A: `{payload.get('config_a_path', 'na')}`",
        f"- Config B: `{payload.get('config_b_path', 'na')}`",
        f"- Seeds: {payload.get('seeds', [])}",
        f"- Authors: {payload.get('n_authors', 'na')}",
        f"- Months: {payload.get('months', 'na')}",
        f"- Difficulty: {payload.get('difficulty', 'na')}",
        "",
        "## Transformer/GRU Compare",
        "",
        "### Final Month Metrics",
        "",
        _build_metric_table(
            metrics_a=final_month_summary_a if isinstance(final_month_summary_a, dict) else {},
            metrics_b=final_month_summary_b if isinstance(final_month_summary_b, dict) else {},
        ),
        "",
        "## Significance (Final Month)",
        "",
        _build_significance_table(final_month_stats if isinstance(final_month_stats, dict) else {}),
        "",
        "## Significance (All Eval Months)",
        "",
        _build_significance_table(all_months_stats if isinstance(all_months_stats, dict) else {}),
        "",
        "## Drift-Performance Correlation",
        "",
        "### A",
        "",
        _build_correlation_table(corr_a.get("summary", {}) if isinstance(corr_a, dict) else {}),
        "",
        "### B",
        "",
        _build_correlation_table(corr_b.get("summary", {}) if isinstance(corr_b, dict) else {}),
    ]

    sweep_path = artifact_root / "eval_temporal_sweep.jsonl"
    if sweep_path.exists():
        sweep_records = _load_jsonl(sweep_path)
        sweep_summary = aggregate_sweep_metrics(sweep_records)
        lines.extend(
            [
                "",
                "## Baseline Sweep Summary",
                "",
                f"- Sweep records: `{sweep_path}`",
            ]
        )
        final_summary = sweep_summary.get("final_month_summary", {})
        all_summary = sweep_summary.get("all_eval_months_summary", {})
        if isinstance(final_summary, dict):
            lines.extend(
                [""] + _build_single_summary_table(title="Final Month", summary=final_summary)
            )
        if isinstance(all_summary, dict):
            lines.extend(
                [""] + _build_single_summary_table(title="All Eval Months", summary=all_summary)
            )

    e2e_compare_path = _find_optional_compare(artifact_root)
    if e2e_compare_path is not None:
        e2e_payload = _load_json(e2e_compare_path)
        e2e_a = e2e_payload.get("final_month_summary_a", {})
        e2e_b = e2e_payload.get("final_month_summary_b", {})
        lines.extend(
            [
                "",
                "## E2E vs Frozen Compare",
                "",
                f"- Compare summary: `{e2e_compare_path}`",
                "",
                _build_metric_table(
                    metrics_a=e2e_a if isinstance(e2e_a, dict) else {},
                    metrics_b=e2e_b if isinstance(e2e_b, dict) else {},
                ),
            ]
        )

    contrastive_metrics = _latest_match(artifact_root, "contrastive/**/pretrain_metrics.json")
    if contrastive_metrics is not None:
        contrastive_payload = _load_json(contrastive_metrics)
        lines.extend(
            [
                "",
                "## Contrastive Pretraining",
                "",
                f"- Metrics: `{contrastive_metrics}`",
                f"- Final loss: {_format_value(contrastive_payload.get('final_loss'))}",
                f"- Pair count: {_format_value(contrastive_payload.get('n_pairs'))}",
                f"- Checkpoint: `{contrastive_payload.get('checkpoint_path', 'na')}`",
            ]
        )

    multitask_ablation = _latest_match(artifact_root, "ablation_drift_weight/ablation_summary.json")
    if multitask_ablation is not None:
        multitask_payload = _load_json(multitask_ablation)
        lines.extend(
            [
                "",
                "## Multitask Drift-Weight Ablation",
                "",
                f"- Summary: `{multitask_ablation}`",
                "",
                "| drift_lambda | accuracy_mean | f1_mean | balanced_accuracy_mean |",
                "|---:|---:|---:|---:|",
            ]
        )
        rows = multitask_payload.get("rows", [])
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "| "
                    f"{_format_value(row.get('drift_lambda'))} | "
                    f"{_format_value(row.get('accuracy_mean'))} | "
                    f"{_format_value(row.get('f1_mean'))} | "
                    f"{_format_value(row.get('balanced_accuracy_mean'))} |"
                )
        lines.append("")
        lines.append(f"- Plot: `{multitask_ablation.parent / 'ablation_drift_weight.png'}`")

    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
            "- Summary A results: "
            f"`{summary_a.get('results_path', 'na') if isinstance(summary_a, dict) else 'na'}`",
            "- Summary B results: "
            f"`{summary_b.get('results_path', 'na') if isinstance(summary_b, dict) else 'na'}`",
        ]
    )

    if isinstance(run_meta_a, list):
        lines.extend(["", "### Run Paths A", ""])
        for row in run_meta_a:
            if not isinstance(row, dict):
                continue
            output_dir = row.get("output_dir", "")
            seed = row.get("seed", "na")
            lines.append(f"- Seed {seed}: `{output_dir}`")
            if output_dir:
                lines.append(f"  - `{output_dir}/per_month_metrics.png`")
                lines.append(f"  - `{output_dir}/threshold_over_time.png`")
                lines.append(f"  - `{output_dir}/pred_rate_over_time.png`")
                lines.append(f"  - `{output_dir}/embedding_drift_over_time.png`")
                lines.append(f"  - `{output_dir}/drift_vs_accuracy_delta.png`")

    if isinstance(run_meta_b, list):
        lines.extend(["", "### Run Paths B", ""])
        for row in run_meta_b:
            if not isinstance(row, dict):
                continue
            output_dir = row.get("output_dir", "")
            seed = row.get("seed", "na")
            lines.append(f"- Seed {seed}: `{output_dir}`")
            if output_dir:
                lines.append(f"  - `{output_dir}/per_month_metrics.png`")
                lines.append(f"  - `{output_dir}/threshold_over_time.png`")
                lines.append(f"  - `{output_dir}/pred_rate_over_time.png`")
                lines.append(f"  - `{output_dir}/embedding_drift_over_time.png`")
                lines.append(f"  - `{output_dir}/drift_vs_accuracy_delta.png`")

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path
