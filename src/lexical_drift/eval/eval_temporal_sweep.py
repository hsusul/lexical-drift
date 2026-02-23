from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from lexical_drift.config import EvalTemporalConfig
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.eval.eval_temporal import run_eval_temporal
from lexical_drift.utils import ensure_dir
from lexical_drift.utils.metadata import config_sha256, git_commit_hash

SUMMARY_METRICS = (
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


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


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


def _summary_stats(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _summarize_entries(
    entries: list[dict[str, object]],
) -> dict[str, dict[str, float | int] | None]:
    summary: dict[str, dict[str, float | int] | None] = {}
    for metric in SUMMARY_METRICS:
        metric_values = []
        for entry in entries:
            value = _as_float(entry.get(metric))
            if value is not None:
                metric_values.append(value)
        summary[metric] = _summary_stats(metric_values)
    return summary


def aggregate_sweep_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    successful = [record for record in records if record.get("status") == "ok"]
    failed = [record for record in records if record.get("status") != "ok"]

    final_entries: list[dict[str, object]] = []
    per_month_entries: list[dict[str, object]] = []
    final_month_indices: list[int] = []

    for record in successful:
        final_entry_raw = record.get("final_month_metrics")
        if isinstance(final_entry_raw, dict):
            final_entry = dict(final_entry_raw)
            final_entries.append(final_entry)
            month_index = final_entry.get("month_index")
            if isinstance(month_index, (int, np.integer)):
                final_month_indices.append(int(month_index))

        month_rows = record.get("per_month")
        if isinstance(month_rows, list):
            for row in month_rows:
                if isinstance(row, dict):
                    per_month_entries.append(dict(row))

    unique_final_months = sorted(set(final_month_indices))
    final_month_index: int | None = (
        unique_final_months[0] if len(unique_final_months) == 1 else None
    )

    return {
        "total_runs": int(len(records)),
        "success_count": int(len(successful)),
        "failure_count": int(len(failed)),
        "failed_seeds": [
            int(record["seed"]) for record in failed if isinstance(record.get("seed"), int)
        ],
        "final_month_index": final_month_index,
        "final_month_summary": _summarize_entries(final_entries),
        "all_eval_months_summary": _summarize_entries(per_month_entries),
    }


def _write_sweep_records_csv(path: Path, records: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "seed": record.get("seed"),
                "status": record.get("status"),
                "model_type": record.get("model_type"),
                "use_time_embeddings": record.get("use_time_embeddings"),
                "loss_type": record.get("loss_type"),
                "pos_weight": record.get("pos_weight"),
                "focal_gamma": record.get("focal_gamma"),
                "final_month_index": record.get("final_month_index"),
                "final_accuracy": record.get("final_accuracy"),
                "final_f1": record.get("final_f1"),
                "final_roc_auc": record.get("final_roc_auc"),
                "final_pr_auc": record.get("final_pr_auc"),
                "chosen_threshold": record.get("chosen_threshold"),
                "final_month_threshold": record.get("final_month_threshold"),
                "cache_fingerprint": record.get("cache_fingerprint"),
                "used_cache": record.get("used_cache"),
                "git_commit_hash": record.get("git_commit_hash"),
                "timestamp_iso": record.get("timestamp_iso"),
                "dataset_hash": record.get("dataset_hash"),
                "config_hash": record.get("config_hash"),
                "metrics_path": record.get("metrics_path"),
                "model_path": record.get("model_path"),
                "output_dir": record.get("output_dir"),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def run_eval_temporal_sweep_with_inputs(
    *,
    config_template: EvalTemporalConfig,
    seeds: list[int],
    seed_input_paths: dict[int, str | Path],
    run_root: str | Path,
    cache_dirs_by_seed: dict[int, str | Path] | None = None,
    results_path: str | Path | None = None,
) -> dict[str, object]:
    output_root = ensure_dir(run_root)
    output_results = (
        Path(results_path) if results_path else output_root / "eval_temporal_sweep.jsonl"
    )
    sweep_metadata_path = output_root / "run_metadata.json"
    output_results.parent.mkdir(parents=True, exist_ok=True)
    if output_results.exists():
        output_results.unlink()

    records: list[dict[str, object]] = []

    for seed in seeds:
        seed_int = int(seed)
        if seed_int not in seed_input_paths:
            raise KeyError(f"Missing input path for seed={seed_int}")
        seed_data_path = Path(seed_input_paths[seed_int])
        seed_output_dir = output_root / f"seed_{seed_int}"
        if cache_dirs_by_seed is not None and seed_int in cache_dirs_by_seed:
            seed_cache_dir = Path(cache_dirs_by_seed[seed_int])
        else:
            seed_cache_dir = seed_output_dir / "cache"

        eval_config = replace(
            config_template,
            input_path=str(seed_data_path),
            output_dir=str(seed_output_dir),
            random_seed=seed_int,
            cache_dir=str(seed_cache_dir),
        )

        try:
            result = run_eval_temporal(eval_config)
            final_month_metrics = dict(result["per_month"][-1])
            thresholds = [float(dict(entry)["threshold_used"]) for entry in result["per_month"]]
            record = {
                "seed": int(seed),
                "status": "ok",
                "model_type": str(result["model_type"]),
                "use_time_embeddings": bool(result.get("use_time_embeddings", True)),
                "loss_type": str(result.get("loss_type", "bce")),
                "pos_weight": result.get("pos_weight"),
                "focal_gamma": float(result.get("focal_gamma", 2.0)),
                "input_path": str(seed_data_path),
                "output_dir": str(seed_output_dir),
                "metrics_path": str(result["metrics_path"]),
                "model_path": str(result["model_path"]),
                "final_month_index": int(result["final_month_index"]),
                "final_accuracy": float(result["final_accuracy"]),
                "final_f1": float(result["final_f1"]),
                "final_roc_auc": _as_float(final_month_metrics.get("roc_auc")),
                "final_pr_auc": _as_float(final_month_metrics.get("pr_auc")),
                "final_month_metrics": final_month_metrics,
                "per_month": [dict(entry) for entry in result["per_month"]],
                "threshold_mode": str(result["threshold_mode"]),
                "calibration_metric": str(result["calibration_metric"]),
                "chosen_threshold": float(result["chosen_threshold"]),
                "final_month_threshold": float(result["final_month_threshold"]),
                "thresholds": thresholds,
                "cache_fingerprint": str(result["cache_fingerprint"]),
                "git_commit_hash": str(result["git_commit_hash"]),
                "timestamp_iso": str(result["timestamp_iso"]),
                "dataset_hash": str(result["dataset_hash"]),
                "config_hash": str(result["config_hash"]),
                "used_cache": bool(result["used_cache"]),
            }
        except Exception as exc:  # pragma: no cover - failure path validated indirectly in summary
            record = {
                "seed": int(seed),
                "status": "error",
                "error": str(exc),
                "input_path": str(seed_data_path),
                "output_dir": str(seed_output_dir),
            }

        _append_jsonl(output_results, record)
        records.append(record)

    aggregate = aggregate_sweep_metrics(records)
    sweep_records_csv_path = output_root / "sweep_records.csv"
    _write_sweep_records_csv(sweep_records_csv_path, records)
    sweep_metadata = {
        "mode": "eval_temporal_sweep",
        "model_type": config_template.model_type,
        "encoder_model": config_template.encoder_model,
        "max_length": int(config_template.max_length),
        "train_months": int(config_template.train_months),
        "seeds": [int(seed) for seed in seeds],
        "results_path": str(output_results),
        "sweep_records_csv_path": str(sweep_records_csv_path),
        "run_root": str(output_root),
        "config_hash": config_sha256(config_template),
        "git_commit_hash": git_commit_hash(),
    }
    sweep_metadata_path.write_text(json.dumps(sweep_metadata, indent=2), encoding="utf-8")
    return {
        "results_path": str(output_results),
        "sweep_records_csv_path": str(sweep_records_csv_path),
        "run_metadata_path": str(sweep_metadata_path),
        "records": records,
        "model_type": config_template.model_type,
        **aggregate,
    }


def run_eval_temporal_sweep(
    *,
    config_template: EvalTemporalConfig,
    seeds: list[int],
    n_authors: int,
    months: int,
    difficulty: str,
    artifact_root: str | Path = "artifacts",
    results_path: str | Path | None = None,
) -> dict[str, object]:
    output_root = ensure_dir(artifact_root)
    seed_inputs: dict[int, Path] = {}
    data_root = output_root / "eval_sweep_data"
    for seed in seeds:
        seed_data_path = data_root / f"synth_seed_{seed}.csv"
        save_synthetic_dataset(
            out_path=seed_data_path,
            n_authors=n_authors,
            months=months,
            random_seed=seed,
            difficulty=difficulty,
        )
        seed_inputs[int(seed)] = seed_data_path

    return run_eval_temporal_sweep_with_inputs(
        config_template=config_template,
        seeds=seeds,
        seed_input_paths=seed_inputs,
        run_root=output_root / "eval_sweep_runs",
        results_path=results_path
        if results_path is not None
        else output_root / "eval_temporal_sweep.jsonl",
    )
