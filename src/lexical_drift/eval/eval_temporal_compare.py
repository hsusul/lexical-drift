from __future__ import annotations

import json
from pathlib import Path

from lexical_drift.config import EvalTemporalConfig
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.eval.eval_temporal_sweep import run_eval_temporal_sweep_with_inputs
from lexical_drift.utils import ensure_dir

METRICS = (
    "accuracy",
    "f1",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "pred_pos_rate",
    "true_pos_rate",
    "threshold_used",
)


def _metric_delta(
    summary_a: dict[str, dict[str, float | int] | None],
    summary_b: dict[str, dict[str, float | int] | None],
) -> dict[str, float | None]:
    delta: dict[str, float | None] = {}
    for metric in METRICS:
        stats_a = summary_a.get(metric)
        stats_b = summary_b.get(metric)
        if not isinstance(stats_a, dict) or not isinstance(stats_b, dict):
            delta[metric] = None
            continue
        if "mean" not in stats_a or "mean" not in stats_b:
            delta[metric] = None
            continue
        delta[metric] = float(stats_b["mean"]) - float(stats_a["mean"])
    return delta


def run_eval_temporal_compare(
    *,
    config_a_template: EvalTemporalConfig,
    config_b_template: EvalTemporalConfig,
    config_a_path: str | Path,
    config_b_path: str | Path,
    seeds: list[int],
    n_authors: int,
    months: int,
    difficulty: str,
    artifact_root: str | Path = "artifacts",
) -> dict[str, object]:
    output_root = ensure_dir(artifact_root)
    data_root = output_root / "compare_data"
    seed_inputs: dict[int, Path] = {}
    shared_cache_dirs: dict[int, Path] = {}
    for seed in seeds:
        seed_path = data_root / f"synth_seed_{seed}.csv"
        save_synthetic_dataset(
            out_path=seed_path,
            n_authors=n_authors,
            months=months,
            random_seed=seed,
            difficulty=difficulty,
        )
        seed_inputs[int(seed)] = seed_path
        shared_cache_dirs[int(seed)] = output_root / "compare_cache" / f"seed_{seed}"

    result_a = run_eval_temporal_sweep_with_inputs(
        config_template=config_a_template,
        seeds=seeds,
        seed_input_paths=seed_inputs,
        run_root=output_root / "compare_runs" / "A",
        cache_dirs_by_seed=shared_cache_dirs,
        results_path=output_root / "compare_runs" / "A" / "eval_temporal_sweep.jsonl",
    )
    result_b = run_eval_temporal_sweep_with_inputs(
        config_template=config_b_template,
        seeds=seeds,
        seed_input_paths=seed_inputs,
        run_root=output_root / "compare_runs" / "B",
        cache_dirs_by_seed=shared_cache_dirs,
        results_path=output_root / "compare_runs" / "B" / "eval_temporal_sweep.jsonl",
    )

    final_month_summary_a = dict(result_a["final_month_summary"])
    final_month_summary_b = dict(result_b["final_month_summary"])
    all_months_summary_a = dict(result_a["all_eval_months_summary"])
    all_months_summary_b = dict(result_b["all_eval_months_summary"])
    final_month_delta = _metric_delta(final_month_summary_a, final_month_summary_b)
    all_months_delta = _metric_delta(all_months_summary_a, all_months_summary_b)

    summary_path = output_root / "eval_temporal_compare_summary.json"
    payload = {
        "config_a_path": str(config_a_path),
        "config_b_path": str(config_b_path),
        "seeds": [int(seed) for seed in seeds],
        "n_authors": int(n_authors),
        "months": int(months),
        "difficulty": difficulty,
        "summary_a": {
            "model_type": str(result_a["model_type"]),
            "total_runs": int(result_a["total_runs"]),
            "success_count": int(result_a["success_count"]),
            "failure_count": int(result_a["failure_count"]),
            "failed_seeds": list(result_a["failed_seeds"]),
            "final_month_index": result_a["final_month_index"],
            "final_month_summary": final_month_summary_a,
            "all_eval_months_summary": all_months_summary_a,
            "results_path": str(result_a["results_path"]),
        },
        "summary_b": {
            "model_type": str(result_b["model_type"]),
            "total_runs": int(result_b["total_runs"]),
            "success_count": int(result_b["success_count"]),
            "failure_count": int(result_b["failure_count"]),
            "failed_seeds": list(result_b["failed_seeds"]),
            "final_month_index": result_b["final_month_index"],
            "final_month_summary": final_month_summary_b,
            "all_eval_months_summary": all_months_summary_b,
            "results_path": str(result_b["results_path"]),
        },
        "final_month_summary_a": final_month_summary_a,
        "final_month_summary_b": final_month_summary_b,
        "final_month_delta": final_month_delta,
        "all_months_summary_a": all_months_summary_a,
        "all_months_summary_b": all_months_summary_b,
        "all_months_delta": all_months_delta,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "summary_path": str(summary_path),
        "final_month_summary_a": final_month_summary_a,
        "final_month_summary_b": final_month_summary_b,
        "final_month_delta": final_month_delta,
        "all_months_summary_a": all_months_summary_a,
        "all_months_summary_b": all_months_summary_b,
        "all_months_delta": all_months_delta,
        "summary_a": payload["summary_a"],
        "summary_b": payload["summary_b"],
    }
