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


def _summarize_final_month(records: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for metric in SUMMARY_METRICS:
        values: list[float] = []
        for record in records:
            if record.get("status") != "ok":
                continue
            final_month = record.get("final_month_metrics")
            if not isinstance(final_month, dict):
                continue
            value = _as_float(final_month.get(metric))
            if value is not None:
                values.append(value)
        summary[metric] = _summary_stats(values)
    return summary


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
                "final_accuracy": record.get("final_accuracy"),
                "final_f1": record.get("final_f1"),
                "final_roc_auc": final_month.get("roc_auc"),
                "final_pr_auc": final_month.get("pr_auc"),
                "use_time_embeddings": record.get("use_time_embeddings"),
                "loss_type": record.get("loss_type"),
                "pos_weight": record.get("pos_weight"),
                "focal_gamma": record.get("focal_gamma"),
                "threshold_mode": record.get("threshold_mode"),
                "calibration_metric": record.get("calibration_metric"),
                "fixed_threshold": record.get("fixed_threshold"),
                "chosen_threshold": record.get("chosen_threshold"),
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
    artifact_root: str | Path = "artifacts",
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
            record: dict[str, object] = {
                "seed": seed_int,
                "status": "ok",
                "input_path": str(seed_data_path),
                "model_type": str(eval_result["model_type"]),
                "final_month_index": int(eval_result["final_month_index"]),
                "final_accuracy": float(eval_result["final_accuracy"]),
                "final_f1": float(eval_result["final_f1"]),
                "final_month_metrics": dict(eval_result["per_month"][-1]),
                "use_time_embeddings": bool(eval_result["use_time_embeddings"]),
                "loss_type": str(eval_result["loss_type"]),
                "pos_weight": eval_result["pos_weight"],
                "focal_gamma": float(eval_result["focal_gamma"]),
                "threshold_mode": str(eval_result["threshold_mode"]),
                "calibration_metric": str(eval_result["calibration_metric"]),
                "fixed_threshold": float(eval_result["fixed_threshold"]),
                "chosen_threshold": float(eval_result["chosen_threshold"]),
                "checkpoint_path": str(eval_result["checkpoint_path"]),
                "model_path": str(eval_result["model_path"]),
                "metrics_path": str(eval_result["metrics_path"]),
                "per_month_csv_path": str(eval_result["per_month_csv_path"]),
                "run_metadata_path": str(eval_result["run_metadata_path"]),
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
    summary = _summarize_final_month(records)
    csv_path = output_root / "e2e_sweep_records.csv"
    _write_e2e_sweep_csv(csv_path, records)

    run_metadata_path = output_root / "run_metadata.json"
    run_metadata = {
        "mode": "eval_e2e_sweep",
        "seeds": [int(seed) for seed in seeds],
        "results_path": str(output_results),
        "records_csv_path": str(csv_path),
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
        "run_metadata_path": str(run_metadata_path),
        "records": records,
        "summary": summary,
        "total_runs": int(len(records)),
        "success_count": int(success_count),
        "failure_count": int(failure_count),
    }
