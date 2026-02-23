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


def _safe_name(value: str) -> str:
    return value.replace("/", "__")


def run_ablation_encoder(
    *,
    config_template: EvalTemporalConfig,
    encoder_models: list[str],
    seeds: list[int],
    n_authors: int,
    months: int,
    difficulty: str,
    artifact_root: str | Path = "artifacts",
) -> dict[str, object]:
    output_root = ensure_dir(artifact_root)
    data_root = output_root / "ablation_encoder_data"
    run_root = output_root / "ablation_encoder_runs"
    cache_root = output_root / "ablation_encoder_cache"

    seed_inputs: dict[int, Path] = {}
    shared_cache_dirs: dict[int, Path] = {}
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
        shared_cache_dirs[seed_int] = cache_root / f"seed_{seed_int}"

    runs: list[dict[str, object]] = []
    rows: list[dict[str, float | str | int | None]] = []
    for encoder_model in encoder_models:
        eval_config = replace(config_template, encoder_model=encoder_model)
        sweep = run_eval_temporal_sweep_with_inputs(
            config_template=eval_config,
            seeds=seeds,
            seed_input_paths=seed_inputs,
            run_root=run_root / _safe_name(encoder_model),
            cache_dirs_by_seed=shared_cache_dirs,
            results_path=run_root / _safe_name(encoder_model) / "eval_temporal_sweep.jsonl",
        )
        final_summary = dict(sweep["final_month_summary"])
        row = {
            "encoder_model": encoder_model,
            "final_accuracy_mean": _safe_mean(final_summary, "accuracy"),
            "final_f1_mean": _safe_mean(final_summary, "f1"),
            "final_balanced_accuracy_mean": _safe_mean(final_summary, "balanced_accuracy"),
            "final_cosine_drift_mean": _safe_mean(final_summary, "cosine_drift"),
            "final_l2_drift_mean": _safe_mean(final_summary, "l2_drift"),
            "final_variance_shift_mean": _safe_mean(final_summary, "variance_shift"),
        }
        rows.append(row)
        runs.append(
            {
                "encoder_model": encoder_model,
                "results_path": str(sweep["results_path"]),
                "final_month_index": sweep["final_month_index"],
                "total_runs": int(sweep["total_runs"]),
                "success_count": int(sweep["success_count"]),
                "failure_count": int(sweep["failure_count"]),
                "final_month_summary": final_summary,
                "all_eval_months_summary": dict(sweep["all_eval_months_summary"]),
            }
        )

    plot_path = output_root / "ablation_encoder.png"
    plt = _import_matplotlib_pyplot()
    x_values = np.arange(len(rows), dtype=np.int64)
    labels = [str(row["encoder_model"]) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    for metric in (
        "final_accuracy_mean",
        "final_f1_mean",
        "final_balanced_accuracy_mean",
    ):
        y_values = np.asarray(
            [
                np.nan if row[metric] is None else float(row[metric])  # type: ignore[index]
                for row in rows
            ],
            dtype=np.float64,
        )
        if np.isnan(y_values).all():
            continue
        ax.plot(x_values, y_values, marker="o", label=metric)
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_xlabel("encoder_model")
    ax.set_ylabel("final month metric mean")
    ax.set_title("Encoder ablation")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    summary_path = output_root / "ablation_encoder_summary.json"
    payload = {
        "config_template": asdict(config_template),
        "encoder_models": encoder_models,
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
