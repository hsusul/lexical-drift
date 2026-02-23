from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


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


def _discover_metric_files(artifact_root: Path) -> list[Path]:
    candidates = sorted(
        list(artifact_root.glob("**/metrics.json"))
        + list(artifact_root.glob("**/eval_temporal_metrics.json"))
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _discover_compare_summaries(artifact_root: Path) -> list[Path]:
    return sorted(artifact_root.glob("**/*compare*summary*.json"))


def _discover_sweep_files(artifact_root: Path) -> list[Path]:
    return sorted(artifact_root.glob("**/*sweep*.jsonl"))


def _extract_final_metrics(payload: dict[str, object]) -> dict[str, object]:
    if isinstance(payload.get("final_month"), dict):
        final = dict(payload["final_month"])
    else:
        final = {}
    for key in ("model_type", "input_path", "train_months", "months_total"):
        if key in payload:
            final[key] = payload[key]
    return final


def _extract_plot_paths(payload: dict[str, object], metrics_path: Path) -> list[Path]:
    paths: list[Path] = []
    plot_paths = payload.get("plot_paths")
    if isinstance(plot_paths, dict):
        for value in plot_paths.values():
            if not isinstance(value, str):
                continue
            path = Path(value)
            if path.exists():
                paths.append(path)
    for name in (
        "per_month_metrics.png",
        "threshold_over_time.png",
        "pred_rate_over_time.png",
        "embedding_drift_over_time.png",
        "drift_vs_accuracy_delta.png",
        "attention_over_time.png",
        "ablation_drift_weight.png",
    ):
        fallback = metrics_path.parent / name
        if fallback.exists():
            paths.append(fallback)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _render_run_compare(artifact_root: Path) -> None:
    metric_files = _discover_metric_files(artifact_root)
    if not metric_files:
        st.info("No metrics files found under artifact root.")
        return

    labels = [str(path.relative_to(artifact_root)) for path in metric_files]
    default_b = 1 if len(labels) > 1 else 0
    run_a_label = st.sidebar.selectbox("Run A", labels, index=0)
    run_b_label = st.sidebar.selectbox("Run B", labels, index=default_b)
    path_a = artifact_root / run_a_label
    path_b = artifact_root / run_b_label

    payload_a = _load_json(path_a)
    payload_b = _load_json(path_b)
    metrics_a = _extract_final_metrics(payload_a)
    metrics_b = _extract_final_metrics(payload_b)

    table = pd.DataFrame(
        [
            {"run": run_a_label, **metrics_a},
            {"run": run_b_label, **metrics_b},
        ]
    )
    st.subheader("Run Comparison")
    st.dataframe(table, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Run A:** `{run_a_label}`")
        for image_path in _extract_plot_paths(payload_a, path_a):
            st.image(str(image_path), caption=image_path.name, use_container_width=True)
    with col_b:
        st.markdown(f"**Run B:** `{run_b_label}`")
        for image_path in _extract_plot_paths(payload_b, path_b):
            st.image(str(image_path), caption=image_path.name, use_container_width=True)


def _render_compare_summary(artifact_root: Path) -> None:
    files = _discover_compare_summaries(artifact_root)
    if not files:
        st.info("No compare summary files found.")
        return
    labels = [str(path.relative_to(artifact_root)) for path in files]
    chosen_label = st.sidebar.selectbox("Compare summary", labels, index=0)
    path = artifact_root / chosen_label
    payload = _load_json(path)
    st.subheader("Compare Summary")
    st.write(f"File: `{path}`")

    final_delta = payload.get("final_month_delta")
    if isinstance(final_delta, dict):
        delta_frame = pd.DataFrame(
            [{"metric": metric, "delta": value} for metric, value in final_delta.items()]
        )
        st.write("Final Month Delta (B - A)")
        st.dataframe(delta_frame, use_container_width=True)

    stats = payload.get("final_month_stats")
    if isinstance(stats, dict):
        rows = []
        for metric, entry in stats.items():
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "metric": metric,
                    "mean_delta": entry.get("mean_delta"),
                    "ci_low": entry.get("ci_low"),
                    "ci_high": entry.get("ci_high"),
                    "p_value": entry.get("p_value"),
                    "n": entry.get("n"),
                }
            )
        if rows:
            st.write("Final Month Significance")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_sweep(artifact_root: Path) -> None:
    files = _discover_sweep_files(artifact_root)
    if not files:
        st.info("No sweep JSONL files found.")
        return
    labels = [str(path.relative_to(artifact_root)) for path in files]
    chosen_label = st.sidebar.selectbox("Sweep file", labels, index=0)
    path = artifact_root / chosen_label
    records = _load_jsonl(path)
    st.subheader("Sweep Records")
    st.write(f"File: `{path}`")
    if not records:
        st.warning("No records found.")
        return

    frame = pd.DataFrame(records)
    st.dataframe(frame, use_container_width=True)
    if "status" in frame.columns:
        ok = frame[frame["status"] == "ok"].copy()
    else:
        ok = frame
    if not ok.empty:
        grouped = ok.groupby("model_type", dropna=False)[["final_accuracy", "final_f1"]].agg(
            ["mean", "std", "min", "max"]
        )
        st.write("Grouped Summary")
        st.dataframe(grouped, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="lexical-drift dashboard", layout="wide")
    st.title("lexical-drift dashboard")

    artifact_root = Path(st.sidebar.text_input("Artifact root", value="artifacts"))
    if not artifact_root.exists():
        st.warning(f"Artifact root not found: {artifact_root}")
        return

    mode = st.sidebar.selectbox(
        "Mode",
        options=["run_compare", "compare_summary", "sweep_jsonl"],
        index=0,
    )
    st.caption(f"Scanning `{artifact_root}`")

    if mode == "run_compare":
        _render_run_compare(artifact_root)
    elif mode == "compare_summary":
        _render_compare_summary(artifact_root)
    else:
        _render_sweep(artifact_root)


if __name__ == "__main__":
    main()
