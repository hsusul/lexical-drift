from __future__ import annotations

from pathlib import Path


def _latest_file(root: Path, pattern: str) -> Path | None:
    matches = [path for path in root.glob(pattern) if path.is_file()]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime_ns)
    return matches[-1]


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _extract_section(text: str, header: str) -> list[str]:
    lines = text.splitlines()
    capture = False
    collected: list[str] = []
    target = f"## {header}"
    for line in lines:
        if line.strip() == target:
            capture = True
            collected.append(line)
            continue
        if capture and line.startswith("## ") and line.strip() != target:
            break
        if capture:
            collected.append(line)
    return collected


def run_render_paper_report(
    *,
    artifact_root: str | Path = "artifacts/experiment_runs",
    out_path: str | Path = "docs/REPORT.md",
) -> dict[str, str]:
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)

    summary_path = _latest_file(root, "**/EXPERIMENT_SUMMARY.md")
    summary_text = _read_text(summary_path)

    index_path = _latest_file(root, "**/INDEX.md")

    ablation_plot = _latest_file(root, "**/ablation_time_embeddings/ablation_delta_plot.png")

    report_lines: list[str] = [
        "# lexical-drift Report",
        "",
        "## Abstract",
        "",
        "This report summarizes the latest offline, reproducible experiment artifacts generated",
        "under `artifacts/experiment_runs`. It is intended for concise presentation of headline",
        "results and ablation outcomes.",
        "",
        "## Artifact Sources",
        "",
        f"- artifact_root: `{root}`",
        f"- experiment_summary: `{summary_path if summary_path else 'na'}`",
        f"- artifact_index: `{index_path if index_path else 'na'}`",
        "",
        "## Headline Metrics",
        "",
    ]

    if summary_text:
        baseline_section = _extract_section(summary_text, "Baseline Sweep (E2E)")
        if baseline_section:
            report_lines.extend(baseline_section)
        else:
            report_lines.append("No baseline sweep section found in experiment summary.")
    else:
        report_lines.append("No experiment summary available.")

    report_lines.extend(["", "## Ablation Highlights", ""])
    if summary_text:
        time_ablation_section = _extract_section(summary_text, "Time-Embedding Ablation")
        loss_section = _extract_section(summary_text, "Loss Ablation Top Configurations")
        if time_ablation_section:
            report_lines.extend(time_ablation_section)
            report_lines.append("")
        if loss_section:
            report_lines.extend(loss_section)
        if not time_ablation_section and not loss_section:
            report_lines.append("No ablation sections found in experiment summary.")
    else:
        report_lines.append("No ablation details available.")

    report_lines.extend(["", "## Figures", ""])
    if ablation_plot is not None:
        report_lines.append(f"![time-ablation]({ablation_plot})")
    else:
        report_lines.append("No plots available.")

    report_lines.extend(["", "## Reproducibility Notes", ""])
    report_lines.append(
        "- Use `lexdrift summarize-experiments` to regenerate summary markdown and best configs."
    )
    report_lines.append("- Use `lexdrift index-artifacts` to refresh artifact path indexing.")

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")

    return {
        "report_path": str(output_path),
        "summary_path": str(summary_path) if summary_path is not None else "",
        "index_path": str(index_path) if index_path is not None else "",
        "artifact_root": str(root),
    }
