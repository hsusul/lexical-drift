from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from lexical_drift.config import EvalE2EConfig, TrainE2EConfig
from lexical_drift.eval.eval_e2e_sweep import run_eval_e2e_sweep
from lexical_drift.eval.stats import bootstrap_ci, paired_t_test
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


def _seed_to_f1(records: list[dict[str, object]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        final_month = record.get("final_month_metrics")
        if not isinstance(final_month, dict):
            continue
        f1 = final_month.get("f1")
        if f1 is None:
            continue
        out[int(record["seed"])] = float(f1)
    return out


def run_ablate_time_embeddings(
    *,
    train_config_template: TrainE2EConfig,
    eval_config_template: EvalE2EConfig,
    seeds: list[int],
    n_authors: int,
    months: int,
    difficulty: str,
    artifact_root: str | Path = "artifacts/experiment_runs",
) -> dict[str, object]:
    output_root = ensure_dir(Path(artifact_root) / "ablation_time_embeddings")

    runs: list[dict[str, object]] = []
    f1_by_setting: dict[str, dict[int, float]] = {}
    for use_time_embeddings in (False, True):
        label = "with_time" if use_time_embeddings else "without_time"
        setting_root = output_root / label
        train_config = replace(
            train_config_template,
            use_time_embeddings=use_time_embeddings,
        )
        sweep_result = run_eval_e2e_sweep(
            train_config_template=train_config,
            eval_config_template=eval_config_template,
            seeds=[int(seed) for seed in seeds],
            n_authors=n_authors,
            months=months,
            difficulty=difficulty,
            artifact_root=setting_root,
            results_path=setting_root / "eval_e2e_sweep.jsonl",
        )
        summary = dict(sweep_result["summary"])
        runs.append(
            {
                "label": label,
                "use_time_embeddings": use_time_embeddings,
                "results_path": str(sweep_result["results_path"]),
                "summary": summary,
                "total_runs": int(sweep_result["total_runs"]),
                "success_count": int(sweep_result["success_count"]),
                "failure_count": int(sweep_result["failure_count"]),
            }
        )
        f1_by_setting[label] = _seed_to_f1(list(sweep_result["records"]))

    shared_seeds = sorted(
        set(f1_by_setting.get("with_time", {}).keys())
        & set(f1_by_setting.get("without_time", {}).keys())
    )
    delta_rows: list[dict[str, object]] = []
    deltas: list[float] = []
    for seed in shared_seeds:
        f1_without = float(f1_by_setting["without_time"][seed])
        f1_with = float(f1_by_setting["with_time"][seed])
        delta = f1_with - f1_without
        deltas.append(delta)
        delta_rows.append(
            {
                "seed": int(seed),
                "f1_without_time": f1_without,
                "f1_with_time": f1_with,
                "delta_f1": float(delta),
            }
        )

    delta_array = np.asarray(deltas, dtype=np.float64)
    t_test = paired_t_test(delta_array)
    ci = bootstrap_ci(delta_array, n_boot=1000, alpha=0.05, seed=0)

    deltas_csv_path = output_root / "ablation_deltas.csv"
    pd.DataFrame(delta_rows).to_csv(deltas_csv_path, index=False)

    plot_path = output_root / "ablation_delta_plot.png"
    plt = _import_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(8, 4))
    x_positions = np.arange(len(delta_rows), dtype=np.int64)
    y_values = np.asarray([float(row["delta_f1"]) for row in delta_rows], dtype=np.float64)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    if y_values.size > 0:
        ax.bar(x_positions, y_values, color="#4C72B0")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(row["seed"]) for row in delta_rows])
    ax.set_xlabel("seed")
    ax.set_ylabel("delta f1 (with_time - without_time)")
    ax.set_title("Time embedding ablation deltas")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    summary_path = output_root / "ablation_summary.json"
    summary_payload = {
        "mode": "ablate_time_embeddings",
        "seeds": [int(seed) for seed in seeds],
        "n_authors": int(n_authors),
        "months": int(months),
        "difficulty": difficulty,
        "train_config_template": asdict(train_config_template),
        "eval_config_template": asdict(eval_config_template),
        "runs": runs,
        "delta_rows": delta_rows,
        "delta_f1_stats": {
            "mean": None if delta_array.size == 0 else float(delta_array.mean()),
            "std": None if delta_array.size == 0 else float(delta_array.std()),
            "min": None if delta_array.size == 0 else float(delta_array.min()),
            "max": None if delta_array.size == 0 else float(delta_array.max()),
            "n": int(delta_array.size),
        },
        "paired_t_test": t_test,
        "bootstrap_ci_95": ci,
        "artifact_paths": {
            "deltas_csv_path": str(deltas_csv_path),
            "ablation_delta_plot_path": str(plot_path),
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "summary_path": str(summary_path),
        "plot_path": str(plot_path),
        "deltas_csv_path": str(deltas_csv_path),
        "paired_t_test": t_test,
        "bootstrap_ci_95": ci,
    }
