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


def _render_compare(path: Path) -> None:
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
        stats_rows = []
        for metric, entry in stats.items():
            if not isinstance(entry, dict):
                continue
            stats_rows.append(
                {
                    "metric": metric,
                    "mean_delta": entry.get("mean_delta"),
                    "ci_low": entry.get("ci_low"),
                    "ci_high": entry.get("ci_high"),
                    "p_value": entry.get("p_value"),
                    "n": entry.get("n"),
                }
            )
        if stats_rows:
            st.write("Final Month Significance")
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

    corr = payload.get("drift_performance_correlation_a")
    if isinstance(corr, dict) and isinstance(corr.get("summary"), dict):
        st.write("Drift/Performance Correlation (A)")
        st.dataframe(
            pd.DataFrame.from_dict(corr["summary"], orient="index"),
            use_container_width=True,
        )


def _render_sweep(path: Path) -> None:
    records = _load_jsonl(path)
    st.subheader("Sweep Records")
    st.write(f"File: `{path}`")
    if not records:
        st.warning("No records found.")
        return

    frame = pd.DataFrame(records)
    st.dataframe(frame, use_container_width=True)
    ok = frame[frame["status"] == "ok"].copy() if "status" in frame else frame
    if not ok.empty:
        grouped = ok.groupby("model_type", dropna=False)[["final_accuracy", "final_f1"]].agg(
            ["mean", "std", "min", "max"]
        )
        st.write("Grouped Summary")
        st.dataframe(grouped, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="lexical-drift dashboard", layout="wide")
    st.title("lexical-drift dashboard")
    mode = st.sidebar.selectbox(
        "Input type",
        options=["compare_json", "sweep_jsonl"],
        index=0,
    )
    default_path = (
        "artifacts/eval_temporal_compare_summary.json"
        if mode == "compare_json"
        else "artifacts/eval_temporal_sweep.jsonl"
    )
    path_text = st.sidebar.text_input("Input path", value=default_path)
    path = Path(path_text)
    if not path.exists():
        st.warning(f"Path not found: {path}")
        return

    if mode == "compare_json":
        _render_compare(path)
    else:
        _render_sweep(path)


if __name__ == "__main__":
    main()
