from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from lexical_drift.config import EvalE2EConfig, TrainE2EConfig
from lexical_drift.eval.eval_e2e_sweep import run_eval_e2e_sweep
from lexical_drift.utils import ensure_dir

_DEF_FOCAL_GAMMA = 2.0


def _build_loss_grid(
    *,
    pos_weights: list[float],
    focal_gammas: list[float],
) -> list[dict[str, float | str | None]]:
    grid: list[dict[str, float | str | None]] = []
    grid.append(
        {
            "loss_label": "bce",
            "loss_type": "bce",
            "pos_weight": None,
            "focal_gamma": _DEF_FOCAL_GAMMA,
        }
    )
    for weight in pos_weights:
        grid.append(
            {
                "loss_label": "weighted_bce",
                "loss_type": "weighted_bce",
                "pos_weight": float(weight),
                "focal_gamma": _DEF_FOCAL_GAMMA,
            }
        )
    for weight in pos_weights:
        for gamma in focal_gammas:
            grid.append(
                {
                    "loss_label": "focal",
                    "loss_type": "focal",
                    "pos_weight": float(weight),
                    "focal_gamma": float(gamma),
                }
            )
    return grid


def _summary_mean(summary: dict[str, object], metric: str) -> float | None:
    raw = summary.get(metric)
    if not isinstance(raw, dict):
        return None
    mean = raw.get("mean")
    if mean is None:
        return None
    return float(mean)


def _best_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    for row in rows:
        score = row.get("f1_mean")
        if score is None:
            continue
        if best is None:
            best = row
            continue
        if float(score) > float(best["f1_mean"]):
            best = row
            continue
        if float(score) == float(best["f1_mean"]):
            current_bal = row.get("balanced_accuracy_mean")
            best_bal = best.get("balanced_accuracy_mean")
            if current_bal is not None and best_bal is not None:
                if float(current_bal) > float(best_bal):
                    best = row
    return best


def run_ablate_loss(
    *,
    train_config_template: TrainE2EConfig,
    eval_config_template: EvalE2EConfig,
    seeds: list[int],
    n_authors: int,
    months: int,
    difficulty: str,
    pos_weights: list[float],
    focal_gammas: list[float],
    artifact_root: str | Path = "artifacts/experiment_runs",
) -> dict[str, object]:
    output_root = ensure_dir(Path(artifact_root) / "ablation_loss")
    grid = _build_loss_grid(pos_weights=pos_weights, focal_gammas=focal_gammas)

    rows: list[dict[str, object]] = []
    for idx, params in enumerate(grid):
        label = str(params["loss_label"])
        loss_type = str(params["loss_type"])
        pos_weight = params["pos_weight"]
        focal_gamma = float(params["focal_gamma"])

        setting_dir = output_root / f"grid_{idx:02d}_{label}"
        train_config = replace(
            train_config_template,
            loss_type=loss_type,
            pos_weight=pos_weight,
            focal_gamma=focal_gamma,
        )
        sweep_result = run_eval_e2e_sweep(
            train_config_template=train_config,
            eval_config_template=eval_config_template,
            seeds=[int(seed) for seed in seeds],
            n_authors=n_authors,
            months=months,
            difficulty=difficulty,
            artifact_root=setting_dir,
            results_path=setting_dir / "eval_e2e_sweep.jsonl",
        )
        summary = dict(sweep_result["summary"])
        row: dict[str, object] = {
            "grid_index": idx,
            "loss_label": label,
            "loss_type": loss_type,
            "pos_weight": pos_weight,
            "focal_gamma": focal_gamma,
            "f1_mean": _summary_mean(summary, "f1"),
            "pr_auc_mean": _summary_mean(summary, "pr_auc"),
            "roc_auc_mean": _summary_mean(summary, "roc_auc"),
            "balanced_accuracy_mean": _summary_mean(summary, "balanced_accuracy"),
            "brier_score_mean": _summary_mean(summary, "brier_score"),
            "ece_mean": _summary_mean(summary, "ece"),
            "chosen_threshold_mean": _summary_mean(summary, "chosen_threshold"),
            "results_path": str(sweep_result["results_path"]),
            "summary_json_path": str(sweep_result["summary_json_path"]),
            "records_csv_path": str(sweep_result["records_csv_path"]),
            "threshold_stability_path": str(sweep_result["threshold_stability_path"]),
            "success_count": int(sweep_result["success_count"]),
            "failure_count": int(sweep_result["failure_count"]),
            "total_runs": int(sweep_result["total_runs"]),
        }
        rows.append(row)

    csv_path = output_root / "loss_grid_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    best = _best_row(rows)
    summary_path = output_root / "summary.json"
    summary_payload = {
        "mode": "ablate_loss",
        "seeds": [int(seed) for seed in seeds],
        "n_authors": int(n_authors),
        "months": int(months),
        "difficulty": difficulty,
        "pos_weights": [float(weight) for weight in pos_weights],
        "focal_gammas": [float(gamma) for gamma in focal_gammas],
        "train_config_template": asdict(train_config_template),
        "eval_config_template": asdict(eval_config_template),
        "rows": rows,
        "best_configuration": best,
        "loss_grid_results_csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "summary_path": str(summary_path),
        "csv_path": str(csv_path),
        "best_configuration": best,
    }
