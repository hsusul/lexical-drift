from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from lexical_drift.config import EvalTemporalConfig
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.eval.eval_temporal_sweep import run_eval_temporal_sweep_with_inputs
from lexical_drift.utils import ensure_dir


def _import_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for ablation plots. Install with: pip install matplotlib"
        ) from exc

    return plt


def _safe_mean(summary: dict[str, object], metric: str) -> float | None:
    raw = summary.get(metric)
    if not isinstance(raw, dict):
        return None
    mean = raw.get("mean")
    if mean is None:
        return None
    return float(mean)


def run_ablation_time_embeddings(
    *,
    config_template: EvalTemporalConfig,
    seeds: list[int],
    n_authors: int,
    months: int,
    difficulty: str,
    artifact_root: str | Path = "artifacts",
) -> dict[str, object]:
    output_root = ensure_dir(Path(artifact_root) / "ablation_time_embeddings")
    data_root = output_root / "data"
    run_root = output_root / "runs"
    cache_root = output_root / "cache"

    seed_inputs: dict[int, Path] = {}
    for seed in seeds:
        seed_int = int(seed)
        seed_path = data_root / f"synth_seed_{seed_int}.csv"
        save_synthetic_dataset(
            out_path=seed_path,
            n_authors=n_authors,
            months=months,
            random_seed=seed_int,
            difficulty=difficulty,
        )
        seed_inputs[seed_int] = seed_path

    rows: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for use_time_embeddings in (False, True):
        cache_dirs_by_seed = {
            int(seed): cache_root / f"time_{int(use_time_embeddings)}" / f"seed_{int(seed)}"
            for seed in seeds
        }
        eval_config = replace(
            config_template,
            model_type="transformer",
            use_time_embeddings=use_time_embeddings,
        )
        sweep = run_eval_temporal_sweep_with_inputs(
            config_template=eval_config,
            seeds=seeds,
            seed_input_paths=seed_inputs,
            run_root=run_root / ("with_time" if use_time_embeddings else "without_time"),
            cache_dirs_by_seed=cache_dirs_by_seed,
            results_path=run_root
            / ("with_time" if use_time_embeddings else "without_time")
            / "eval_temporal_sweep.jsonl",
        )
        final_summary = dict(sweep["final_month_summary"])
        row = {
            "use_time_embeddings": bool(use_time_embeddings),
            "final_accuracy_mean": _safe_mean(final_summary, "accuracy"),
            "final_f1_mean": _safe_mean(final_summary, "f1"),
            "final_balanced_accuracy_mean": _safe_mean(final_summary, "balanced_accuracy"),
        }
        rows.append(row)
        runs.append(
            {
                "use_time_embeddings": bool(use_time_embeddings),
                "results_path": str(sweep["results_path"]),
                "final_month_index": sweep["final_month_index"],
                "total_runs": int(sweep["total_runs"]),
                "success_count": int(sweep["success_count"]),
                "failure_count": int(sweep["failure_count"]),
                "final_month_summary": final_summary,
                "all_eval_months_summary": dict(sweep["all_eval_months_summary"]),
            }
        )

    plot_path = output_root / "ablation_time_embeddings.png"
    plt = _import_matplotlib_pyplot()
    labels = ["without_time", "with_time"]
    x_values = np.arange(len(labels), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(8, 4))
    for metric in (
        "final_accuracy_mean",
        "final_f1_mean",
        "final_balanced_accuracy_mean",
    ):
        y_values = np.asarray(
            [np.nan if row[metric] is None else float(row[metric]) for row in rows],
            dtype=np.float64,
        )
        if np.isnan(y_values).all():
            continue
        ax.plot(x_values, y_values, marker="o", label=metric)
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_xlabel("setting")
    ax.set_ylabel("final month metric mean")
    ax.set_title("Transformer time embedding ablation")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    summary_path = output_root / "ablation_summary.json"
    payload = {
        "config_template": asdict(config_template),
        "seeds": [int(seed) for seed in seeds],
        "n_authors": int(n_authors),
        "months": int(months),
        "difficulty": difficulty,
        "rows": rows,
        "runs": runs,
        "plot_path": str(plot_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "summary_path": str(summary_path),
        "plot_path": str(plot_path),
        "rows": rows,
        "runs": runs,
    }
