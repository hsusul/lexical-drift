from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lexical_drift.config import EvalTemporalConfig
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.eval.eval_temporal_sweep import run_eval_temporal_sweep_with_inputs
from lexical_drift.eval.stats import bootstrap_ci, paired_t_test
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
    "cosine_drift",
    "l2_drift",
    "variance_shift",
    "accuracy_delta_from_ref",
    "f1_delta_from_ref",
)

DRIFT_METRICS = ("cosine_drift", "l2_drift", "variance_shift")
PERF_DELTA_METRICS = ("accuracy_delta_from_ref", "f1_delta_from_ref")


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(cast):
        return None
    return cast


def _empty_stat_record() -> dict[str, float | int | None]:
    return {
        "mean_delta": None,
        "ci_low": None,
        "ci_high": None,
        "t_stat": None,
        "p_value": None,
        "n": 0,
    }


def _metric_mean_from_per_month(record: dict[str, object], metric: str) -> float | None:
    per_month = record.get("per_month")
    if not isinstance(per_month, list) or not per_month:
        return None
    values: list[float] = []
    for entry in per_month:
        if not isinstance(entry, dict):
            return None
        value = _as_float(entry.get(metric))
        if value is None:
            return None
        values.append(value)
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _pearson_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std <= 0.0 or y_std <= 0.0:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return None
    return corr


def _build_seed_metric_arrays(
    *,
    records_a: dict[int, dict[str, object]],
    records_b: dict[int, dict[str, object]],
    seeds: list[int],
    metric: str,
    use_final_month: bool,
) -> dict[str, list[float] | list[int]]:
    output: dict[str, list[float] | list[int]] = {
        "seeds": [],
        "a_values": [],
        "b_values": [],
        "delta_values": [],
    }

    for seed in seeds:
        record_a = records_a.get(int(seed))
        record_b = records_b.get(int(seed))
        if record_a is None or record_b is None:
            return output

        if use_final_month:
            final_a = record_a.get("final_month_metrics")
            final_b = record_b.get("final_month_metrics")
            if not isinstance(final_a, dict) or not isinstance(final_b, dict):
                return output
            value_a = _as_float(final_a.get(metric))
            value_b = _as_float(final_b.get(metric))
        else:
            value_a = _metric_mean_from_per_month(record_a, metric)
            value_b = _metric_mean_from_per_month(record_b, metric)

        if value_a is None or value_b is None:
            return output

        output["seeds"].append(int(seed))
        output["a_values"].append(value_a)
        output["b_values"].append(value_b)
        output["delta_values"].append(float(value_b - value_a))

    return output


def _stats_from_delta_values(
    delta_values: list[float],
    *,
    seed: int,
) -> dict[str, float | int | None]:
    if not delta_values:
        return _empty_stat_record()
    deltas = np.asarray(delta_values, dtype=np.float64)
    ci = bootstrap_ci(deltas, n_boot=2000, alpha=0.05, seed=seed)
    t_test = paired_t_test(deltas)
    return {
        "mean_delta": ci["mean"],
        "ci_low": ci["low"],
        "ci_high": ci["high"],
        "t_stat": t_test["t_stat"],
        "p_value": t_test["p_value"],
        "n": int(ci["n"]) if ci["n"] is not None else 0,
    }


def _compute_paired_stats(
    *,
    result_a: dict[str, object],
    result_b: dict[str, object],
    seeds: list[int],
) -> tuple[
    dict[str, dict[str, list[float] | list[int]]],
    dict[str, dict[str, float | int | None]],
    dict[str, dict[str, list[float] | list[int]]],
    dict[str, dict[str, float | int | None]],
]:
    records_a = {
        int(record["seed"]): record
        for record in result_a["records"]
        if isinstance(record, dict)
        and record.get("status") == "ok"
        and isinstance(record.get("seed"), int)
    }
    records_b = {
        int(record["seed"]): record
        for record in result_b["records"]
        if isinstance(record, dict)
        and record.get("status") == "ok"
        and isinstance(record.get("seed"), int)
    }

    final_month_values: dict[str, dict[str, list[float] | list[int]]] = {}
    final_month_stats: dict[str, dict[str, float | int | None]] = {}
    all_months_values: dict[str, dict[str, list[float] | list[int]]] = {}
    all_months_stats: dict[str, dict[str, float | int | None]] = {}

    for metric in METRICS:
        final_values = _build_seed_metric_arrays(
            records_a=records_a,
            records_b=records_b,
            seeds=seeds,
            metric=metric,
            use_final_month=True,
        )
        all_values = _build_seed_metric_arrays(
            records_a=records_a,
            records_b=records_b,
            seeds=seeds,
            metric=metric,
            use_final_month=False,
        )
        final_month_values[metric] = final_values
        all_months_values[metric] = all_values
        final_month_stats[metric] = _stats_from_delta_values(
            final_values["delta_values"],
            seed=0,
        )
        all_months_stats[metric] = _stats_from_delta_values(
            all_values["delta_values"],
            seed=1,
        )

    return final_month_values, final_month_stats, all_months_values, all_months_stats


def _compute_drift_perf_correlations(
    result: dict[str, object],
) -> dict[str, object]:
    pair_keys = [
        f"{drift_metric}__{perf_metric}"
        for drift_metric in DRIFT_METRICS
        for perf_metric in PERF_DELTA_METRICS
    ]
    per_seed: list[dict[str, object]] = []
    summary_values: dict[str, list[float]] = {key: [] for key in pair_keys}

    for record in result["records"]:
        if not isinstance(record, dict):
            continue
        if record.get("status") != "ok" or not isinstance(record.get("seed"), int):
            continue
        per_month = record.get("per_month")
        if not isinstance(per_month, list):
            continue

        correlations: dict[str, float | None] = {}
        for drift_metric in DRIFT_METRICS:
            x_values: list[float] = []
            for entry in per_month:
                if not isinstance(entry, dict):
                    x_values = []
                    break
                value = _as_float(entry.get(drift_metric))
                if value is None:
                    x_values = []
                    break
                x_values.append(value)

            for perf_metric in PERF_DELTA_METRICS:
                y_values: list[float] = []
                for entry in per_month:
                    if not isinstance(entry, dict):
                        y_values = []
                        break
                    value = _as_float(entry.get(perf_metric))
                    if value is None:
                        y_values = []
                        break
                    y_values.append(value)

                pair_key = f"{drift_metric}__{perf_metric}"
                if not x_values or not y_values:
                    correlations[pair_key] = None
                    continue
                corr = _pearson_correlation(x_values, y_values)
                correlations[pair_key] = corr
                if corr is not None:
                    summary_values[pair_key].append(corr)

        per_seed.append({"seed": int(record["seed"]), "correlations": correlations})

    summary: dict[str, dict[str, float | int] | None] = {}
    for pair_key in pair_keys:
        values = summary_values[pair_key]
        if not values:
            summary[pair_key] = None
            continue
        array = np.asarray(values, dtype=np.float64)
        summary[pair_key] = {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
            "n": int(array.size),
        }
    return {"per_seed": per_seed, "summary": summary}


def _extract_run_metadata(records: list[dict[str, object]]) -> list[dict[str, object]]:
    metadata_rows: list[dict[str, object]] = []
    for record in records:
        if record.get("status") != "ok":
            continue
        metadata_rows.append(
            {
                "seed": int(record["seed"]),
                "input_path": str(record["input_path"]),
                "output_dir": str(record["output_dir"]),
                "git_commit_hash": str(record.get("git_commit_hash", "unknown")),
                "timestamp_iso": str(record.get("timestamp_iso", "")),
                "dataset_hash": str(record.get("dataset_hash", "")),
                "config_hash": str(record.get("config_hash", "")),
            }
        )
    return metadata_rows


def _write_compare_seed_deltas_csv(
    *,
    path: Path,
    final_month_values: dict[str, dict[str, list[float] | list[int]]],
    all_months_values: dict[str, dict[str, list[float] | list[int]]],
) -> None:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        final_values = final_month_values.get(metric, {})
        all_values = all_months_values.get(metric, {})
        final_seed_map: dict[int, tuple[float, float, float]] = {}
        all_seed_map: dict[int, tuple[float, float, float]] = {}

        final_seeds = final_values.get("seeds", [])
        final_a_values = final_values.get("a_values", [])
        final_b_values = final_values.get("b_values", [])
        final_delta_values = final_values.get("delta_values", [])
        for idx, seed in enumerate(final_seeds):
            final_seed_map[int(seed)] = (
                float(final_a_values[idx]),
                float(final_b_values[idx]),
                float(final_delta_values[idx]),
            )

        all_seeds = all_values.get("seeds", [])
        all_a_values = all_values.get("a_values", [])
        all_b_values = all_values.get("b_values", [])
        all_delta_values = all_values.get("delta_values", [])
        for idx, seed in enumerate(all_seeds):
            all_seed_map[int(seed)] = (
                float(all_a_values[idx]),
                float(all_b_values[idx]),
                float(all_delta_values[idx]),
            )

        metric_seeds = sorted(set(final_seed_map) | set(all_seed_map))
        for seed in metric_seeds:
            final_triplet = final_seed_map.get(seed, (None, None, None))
            all_triplet = all_seed_map.get(seed, (None, None, None))
            rows.append(
                {
                    "seed": int(seed),
                    "metric": metric,
                    "final_a": final_triplet[0],
                    "final_b": final_triplet[1],
                    "final_delta": final_triplet[2],
                    "all_months_a": all_triplet[0],
                    "all_months_b": all_triplet[1],
                    "all_months_delta": all_triplet[2],
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)


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
    (
        final_month_values,
        final_month_stats,
        all_months_values,
        all_months_stats,
    ) = _compute_paired_stats(result_a=result_a, result_b=result_b, seeds=seeds)
    drift_perf_correlation_a = _compute_drift_perf_correlations(result_a)
    drift_perf_correlation_b = _compute_drift_perf_correlations(result_b)
    per_run_metadata_a = _extract_run_metadata(result_a["records"])
    per_run_metadata_b = _extract_run_metadata(result_b["records"])
    compare_seed_deltas_csv_path = output_root / "compare_seed_deltas.csv"
    _write_compare_seed_deltas_csv(
        path=compare_seed_deltas_csv_path,
        final_month_values=final_month_values,
        all_months_values=all_months_values,
    )

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
            "drift_performance_correlation": drift_perf_correlation_a,
            "per_run_metadata": per_run_metadata_a,
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
            "drift_performance_correlation": drift_perf_correlation_b,
            "per_run_metadata": per_run_metadata_b,
        },
        "final_month_summary_a": final_month_summary_a,
        "final_month_summary_b": final_month_summary_b,
        "final_month_delta": final_month_delta,
        "final_month_values": final_month_values,
        "final_month_stats": final_month_stats,
        "all_months_summary_a": all_months_summary_a,
        "all_months_summary_b": all_months_summary_b,
        "all_months_delta": all_months_delta,
        "all_months_values": all_months_values,
        "all_months_stats": all_months_stats,
        "drift_performance_correlation_a": drift_perf_correlation_a,
        "drift_performance_correlation_b": drift_perf_correlation_b,
        "per_run_metadata_a": per_run_metadata_a,
        "per_run_metadata_b": per_run_metadata_b,
        "compare_seed_deltas_csv_path": str(compare_seed_deltas_csv_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "summary_path": str(summary_path),
        "final_month_summary_a": final_month_summary_a,
        "final_month_summary_b": final_month_summary_b,
        "final_month_delta": final_month_delta,
        "final_month_values": final_month_values,
        "final_month_stats": final_month_stats,
        "all_months_summary_a": all_months_summary_a,
        "all_months_summary_b": all_months_summary_b,
        "all_months_delta": all_months_delta,
        "all_months_values": all_months_values,
        "all_months_stats": all_months_stats,
        "drift_performance_correlation_a": drift_perf_correlation_a,
        "drift_performance_correlation_b": drift_perf_correlation_b,
        "per_run_metadata_a": per_run_metadata_a,
        "per_run_metadata_b": per_run_metadata_b,
        "compare_seed_deltas_csv_path": str(compare_seed_deltas_csv_path),
        "summary_a": payload["summary_a"],
        "summary_b": payload["summary_b"],
    }
