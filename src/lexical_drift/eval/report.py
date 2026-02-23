from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def _format_value(value: object) -> str:
    if value is None:
        return "na"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


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


def render_compare_report(
    *,
    compare_summary_path: str | Path,
    out_path: str | Path,
) -> Path:
    summary_path = Path(compare_summary_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

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
        f"- Compare summary: `{summary_path}`",
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
        "## Final Month Metrics",
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
        "",
        "## Artifact Paths",
        "",
        "- Summary A results: "
        f"`{summary_a.get('results_path', 'na') if isinstance(summary_a, dict) else 'na'}`",
        "- Summary B results: "
        f"`{summary_b.get('results_path', 'na') if isinstance(summary_b, dict) else 'na'}`",
    ]

    if isinstance(run_meta_a, list):
        lines.append("")
        lines.append("### Run Paths A")
        lines.append("")
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
        lines.append("")
        lines.append("### Run Paths B")
        lines.append("")
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
