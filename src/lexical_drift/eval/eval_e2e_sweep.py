from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from lexical_drift.config import EvalE2EConfig, TrainE2EConfig
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.train.e2e_temporal import run_eval_e2e, run_train_e2e
from lexical_drift.utils import ensure_dir
from lexical_drift.utils.metadata import config_sha256, git_commit_hash

SUMMARY_METRICS = (
    "f1",
    "pr_auc",
    "roc_auc",
    "balanced_accuracy",
    "chosen_threshold",
    "brier_score",
    "ece",
)

FINAL_MONTH_METRIC_MAP = {
    "f1": "f1",
    "pr_auc": "pr_auc",
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy",
    "brier_score": "brier_score",
    "ece": "ece",
}


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


def _summary_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _summarize_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    per_metric: dict[str, object] = {}
    for metric in SUMMARY_METRICS:
        values: list[float] = []
        source_key = FINAL_MONTH_METRIC_MAP.get(metric)
        for record in records:
            if record.get("status") != "ok":
                continue
            if source_key is None:
                value = _as_float(record.get(metric))
            else:
                final_month = record.get("final_month_metrics")
                if not isinstance(final_month, dict):
                    value = None
                else:
                    value = _as_float(final_month.get(source_key))
            if value is not None:
                values.append(value)
        per_metric[metric] = _summary_stats(values)
    return per_metric


def _safe_corrcoef(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if np.allclose(float(x.std()), 0.0) or np.allclose(float(y.std()), 0.0):
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return None
    return corr


def _compute_threshold_stability(records: list[dict[str, object]]) -> dict[str, object]:
    chosen_thresholds: list[float] = []
    threshold_var_by_seed: list[dict[str, float | int]] = []
    f1_var_by_seed: list[dict[str, float | int]] = []
    threshold_var_values: list[float] = []
    f1_var_values: list[float] = []

    for record in records:
        if record.get("status") != "ok":
            continue
        seed = int(record["seed"])
        chosen = _as_float(record.get("chosen_threshold"))
        if chosen is not None:
            chosen_thresholds.append(chosen)

        per_month = record.get("per_month")
        if not isinstance(per_month, list) or not per_month:
            continue

        threshold_values: list[float] = []
        f1_values: list[float] = []
        for entry in per_month:
            if not isinstance(entry, dict):
                continue
            threshold_value = _as_float(entry.get("threshold_used"))
            f1_value = _as_float(entry.get("f1"))
            if threshold_value is not None:
                threshold_values.append(threshold_value)
            if f1_value is not None:
                f1_values.append(f1_value)

        if threshold_values:
            threshold_var = float(np.var(np.asarray(threshold_values, dtype=np.float64)))
            threshold_var_by_seed.append({"seed": seed, "variance": threshold_var})
        if f1_values:
            f1_var = float(np.var(np.asarray(f1_values, dtype=np.float64)))
            f1_var_by_seed.append({"seed": seed, "variance": f1_var})
        if threshold_values and f1_values:
            threshold_var_values.append(
                float(np.var(np.asarray(threshold_values, dtype=np.float64)))
            )
            f1_var_values.append(float(np.var(np.asarray(f1_values, dtype=np.float64))))

    return {
        "chosen_threshold_variance": (
            float(np.var(np.asarray(chosen_thresholds, dtype=np.float64)))
            if chosen_thresholds
            else None
        ),
        "threshold_variance_by_seed": threshold_var_by_seed,
        "f1_variance_by_seed": f1_var_by_seed,
        "threshold_f1_variance_correlation": _safe_corrcoef(
            threshold_var_values,
            f1_var_values,
        ),
    }


def _write_e2e_sweep_csv(path: Path, records: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for record in records:
        final_month = record.get("final_month_metrics", {})
        final_month = final_month if isinstance(final_month, dict) else {}
        rows.append(
            {
                "seed": record.get("seed"),
                "status": record.get("status"),
                "final_month_index": record.get("final_month_index"),
                "f1": final_month.get("f1"),
                "pr_auc": final_month.get("pr_auc"),
                "roc_auc": final_month.get("roc_auc"),
                "balanced_accuracy": final_month.get("balanced_accuracy"),
                "brier_score": final_month.get("brier_score"),
                "ece": final_month.get("ece"),
                "chosen_threshold": record.get("chosen_threshold"),
                "final_month_threshold_used": record.get("final_month_threshold_used"),
                "use_time_embeddings": record.get("use_time_embeddings"),
                "loss_type": record.get("loss_type"),
                "pos_weight": record.get("pos_weight"),
                "focal_gamma": record.get("focal_gamma"),
                "threshold_mode": record.get("threshold_mode"),
                "calibration_metric": record.get("calibration_metric"),
                "fixed_threshold": record.get("fixed_threshold"),
                "checkpoint_path": record.get("checkpoint_path"),
                "metrics_path": record.get("metrics_path"),
                "model_path": record.get("model_path"),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def run_eval_e2e_sweep(
    *,
    train_config_template: TrainE2EConfig,
    eval_config_template: EvalE2EConfig,
    seeds: list[int],
    n_authors: int,
    months: int,
    difficulty: str,
    artifact_root: str | Path = "artifacts/experiment_runs",
    results_path: str | Path | None = None,
) -> dict[str, object]:
    output_root = ensure_dir(Path(artifact_root) / "e2e_sweep_runs")
    data_root = ensure_dir(Path(artifact_root) / "e2e_sweep_data")
    output_results = (
        Path(results_path) if results_path else (Path(artifact_root) / "eval_e2e_sweep.jsonl")
    )
    output_results.parent.mkdir(parents=True, exist_ok=True)
    if output_results.exists():
        output_results.unlink()

    records: list[dict[str, object]] = []
    for seed in seeds:
        seed_int = int(seed)
        seed_data_path = data_root / f"synth_seed_{seed_int}.csv"
        save_synthetic_dataset(
            out_path=seed_data_path,
            n_authors=n_authors,
            months=months,
            random_seed=seed_int,
            difficulty=difficulty,
        )

        train_output = output_root / f"seed_{seed_int}" / "train"
        eval_output = output_root / f"seed_{seed_int}" / "eval"
        train_config = replace(
            train_config_template,
            input_path=str(seed_data_path),
            output_dir=str(train_output),
            random_seed=seed_int,
        )
        try:
            train_result = run_train_e2e(train_config)
            eval_config = replace(
                eval_config_template,
                input_path=str(seed_data_path),
                output_dir=str(eval_output),
                random_seed=seed_int,
                checkpoint_path=str(train_result["model_path"]),
            )
            eval_result = run_eval_e2e(eval_config)
            final_month_metrics = dict(eval_result["per_month"][-1])
            final_threshold_used = _as_float(final_month_metrics.get("threshold_used"))
            if final_threshold_used is None:
                final_threshold_used = float(eval_result["chosen_threshold"])
            final_month_metrics["threshold_used"] = float(final_threshold_used)

            record: dict[str, object] = {
                "seed": seed_int,
                "status": "ok",
                "input_path": str(seed_data_path),
                "model_type": str(eval_result["model_type"]),
                "final_month_index": int(eval_result["final_month_index"]),
                "final_accuracy": float(eval_result["final_accuracy"]),
                "final_f1": float(eval_result["final_f1"]),
                "final_month_metrics": final_month_metrics,
                "use_time_embeddings": bool(eval_result["use_time_embeddings"]),
                "loss_type": str(eval_result["loss_type"]),
                "pos_weight": eval_result["pos_weight"],
                "focal_gamma": float(eval_result["focal_gamma"]),
                "threshold_mode": str(eval_result["threshold_mode"]),
                "calibration_metric": str(eval_result["calibration_metric"]),
                "fixed_threshold": float(eval_result["fixed_threshold"]),
                "chosen_threshold": float(eval_result["chosen_threshold"]),
                "final_month_threshold_used": float(final_threshold_used),
                "checkpoint_path": str(eval_result["checkpoint_path"]),
                "model_path": str(eval_result["model_path"]),
                "metrics_path": str(eval_result["metrics_path"]),
                "per_month_csv_path": str(eval_result["per_month_csv_path"]),
                "run_metadata_path": str(eval_result["run_metadata_path"]),
                "per_month": [dict(entry) for entry in eval_result["per_month"]],
            }
        except Exception as exc:  # pragma: no cover
            record = {
                "seed": seed_int,
                "status": "error",
                "input_path": str(seed_data_path),
                "error": str(exc),
            }
        _append_jsonl(output_results, record)
        records.append(record)

    success_count = sum(1 for record in records if record.get("status") == "ok")
    failure_count = len(records) - success_count
    per_metric = _summarize_metrics(records)
    summary_payload = {
        "per_metric": per_metric,
        "total_runs": int(len(records)),
        "success_count": int(success_count),
        "failure_count": int(failure_count),
    }

    csv_path = output_root / "e2e_sweep_records.csv"
    _write_e2e_sweep_csv(csv_path, records)

    summary_json_path = output_root / "e2e_sweep_summary.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    threshold_stability = _compute_threshold_stability(records)
    threshold_stability_path = output_root / "threshold_stability.json"
    threshold_stability_path.write_text(
        json.dumps(threshold_stability, indent=2),
        encoding="utf-8",
    )

    run_metadata_path = output_root / "run_metadata.json"
    run_metadata = {
        "mode": "eval_e2e_sweep",
        "seeds": [int(seed) for seed in seeds],
        "results_path": str(output_results),
        "records_csv_path": str(csv_path),
        "summary_json_path": str(summary_json_path),
        "threshold_stability_path": str(threshold_stability_path),
        "total_runs": int(len(records)),
        "success_count": int(success_count),
        "failure_count": int(failure_count),
        "train_config_hash": config_sha256(train_config_template),
        "eval_config_hash": config_sha256(eval_config_template),
        "git_commit_hash": git_commit_hash(),
    }
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    return {
        "results_path": str(output_results),
        "records_csv_path": str(csv_path),
        "summary_json_path": str(summary_json_path),
        "threshold_stability_path": str(threshold_stability_path),
        "run_metadata_path": str(run_metadata_path),
        "records": records,
        "summary": per_metric,
        "threshold_stability": threshold_stability,
        "total_runs": int(len(records)),
        "success_count": int(success_count),
        "failure_count": int(failure_count),
    }
