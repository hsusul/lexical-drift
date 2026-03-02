from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lexical_drift.config import EvalE2EConfig, TrainE2EConfig
from lexical_drift.eval import ablate_loss_e2e, ablate_time_embeddings_e2e


def _build_train_template(tmp_path: Path) -> TrainE2EConfig:
    return TrainE2EConfig(
        input_path=str(tmp_path / "placeholder.csv"),
        output_dir=str(tmp_path / "artifacts"),
        random_seed=7,
        encoder_model="fake",
        max_length=32,
        batch_size=4,
        train_months=3,
        gru_hidden_dim=16,
        gru_layers=1,
        dropout=0.1,
        lr=0.001,
        epochs=1,
        test_size=0.25,
        pooling="cls",
        freeze_encoder=False,
        pretrained_encoder_path="",
        use_time_embeddings=True,
        loss_type="bce",
        pos_weight=None,
        focal_gamma=2.0,
    )


def _build_eval_template(tmp_path: Path) -> EvalE2EConfig:
    return EvalE2EConfig(
        input_path=str(tmp_path / "placeholder.csv"),
        output_dir=str(tmp_path / "artifacts"),
        random_seed=7,
        encoder_model="fake",
        max_length=32,
        batch_size=4,
        train_months=3,
        test_size=0.25,
        checkpoint_path="",
        pooling="cls",
        threshold=0.5,
        threshold_mode="calibrate_on_val",
        calibration_metric="balanced_accuracy",
        fixed_threshold=0.5,
        threshold_min=0.05,
        threshold_max=0.95,
        n_thresholds=101,
        pretrained_encoder_path="",
    )


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _fake_run_eval_e2e_sweep(**kwargs):
    train_config = kwargs["train_config_template"]
    seeds = [int(seed) for seed in kwargs["seeds"]]
    artifact_root = Path(kwargs["artifact_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)

    f1_values: list[float] = []
    pr_values: list[float] = []
    roc_values: list[float] = []
    bal_values: list[float] = []
    thr_values: list[float] = []
    brier_values: list[float] = []
    ece_values: list[float] = []
    records: list[dict[str, object]] = []

    for seed in seeds:
        base = 0.55 + 0.02 * (seed % 3)
        if bool(train_config.use_time_embeddings):
            base += 0.05
        if train_config.loss_type == "weighted_bce":
            base += 0.01 * float(train_config.pos_weight or 1.0)
        elif train_config.loss_type == "focal":
            base += 0.01 * float(train_config.pos_weight or 1.0)
            base += 0.005 * float(train_config.focal_gamma)

        f1 = float(np.clip(base, 0.0, 0.99))
        pr_auc = float(np.clip(f1 + 0.05, 0.0, 1.0))
        roc_auc = float(np.clip(f1 + 0.07, 0.0, 1.0))
        balanced = float(np.clip(f1 - 0.03, 0.0, 1.0))
        threshold = float(0.45 + 0.01 * (seed % 2))
        brier = float(np.clip(0.4 - f1 * 0.2, 0.0, 1.0))
        ece = float(np.clip(0.3 - f1 * 0.1, 0.0, 1.0))

        f1_values.append(f1)
        pr_values.append(pr_auc)
        roc_values.append(roc_auc)
        bal_values.append(balanced)
        thr_values.append(threshold)
        brier_values.append(brier)
        ece_values.append(ece)

        records.append(
            {
                "seed": seed,
                "status": "ok",
                "final_month_metrics": {
                    "f1": f1,
                    "pr_auc": pr_auc,
                    "roc_auc": roc_auc,
                    "balanced_accuracy": balanced,
                    "brier_score": brier,
                    "ece": ece,
                    "threshold_used": threshold,
                },
                "chosen_threshold": threshold,
                "final_month_threshold_used": threshold,
                "per_month": [
                    {"month_index": 3, "f1": f1 - 0.02, "threshold_used": threshold - 0.01},
                    {"month_index": 4, "f1": f1, "threshold_used": threshold},
                ],
            }
        )

    summary = {
        "f1": _stats(f1_values),
        "pr_auc": _stats(pr_values),
        "roc_auc": _stats(roc_values),
        "balanced_accuracy": _stats(bal_values),
        "chosen_threshold": _stats(thr_values),
        "brier_score": _stats(brier_values),
        "ece": _stats(ece_values),
    }

    results_path = Path(kwargs["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("", encoding="utf-8")
    records_csv_path = artifact_root / "records.csv"
    pd.DataFrame({"seed": seeds, "f1": f1_values}).to_csv(records_csv_path, index=False)
    summary_json_path = artifact_root / "summary.json"
    summary_json_path.write_text(json.dumps({"per_metric": summary}, indent=2), encoding="utf-8")
    threshold_path = artifact_root / "threshold_stability.json"
    threshold_path.write_text(
        json.dumps(
            {
                "chosen_threshold_variance": float(np.var(np.asarray(thr_values))),
                "threshold_f1_variance_correlation": 0.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "results_path": str(results_path),
        "records_csv_path": str(records_csv_path),
        "summary_json_path": str(summary_json_path),
        "threshold_stability_path": str(threshold_path),
        "run_metadata_path": str(artifact_root / "run_metadata.json"),
        "records": records,
        "summary": summary,
        "total_runs": len(seeds),
        "success_count": len(seeds),
        "failure_count": 0,
        "threshold_stability": {
            "chosen_threshold_variance": float(np.var(np.asarray(thr_values))),
            "threshold_f1_variance_correlation": 0.0,
        },
    }


def test_run_ablate_time_embeddings_outputs_summary(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        ablate_time_embeddings_e2e,
        "run_eval_e2e_sweep",
        _fake_run_eval_e2e_sweep,
    )
    result = ablate_time_embeddings_e2e.run_ablate_time_embeddings(
        train_config_template=_build_train_template(tmp_path),
        eval_config_template=_build_eval_template(tmp_path),
        seeds=[1, 2],
        n_authors=20,
        months=6,
        difficulty="hard",
        artifact_root=tmp_path / "artifacts" / "experiment_runs",
    )

    summary_path = Path(str(result["summary_path"]))
    assert summary_path.exists()
    assert Path(str(result["plot_path"])).exists()
    assert Path(str(result["deltas_csv_path"])).exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "ablate_time_embeddings"
    assert isinstance(payload["paired_t_test"]["n"], int)
    assert isinstance(payload["bootstrap_ci_95"]["mean"], float)
    assert len(payload["delta_rows"]) == 2


def test_run_ablate_loss_outputs_grid(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        ablate_loss_e2e,
        "run_eval_e2e_sweep",
        _fake_run_eval_e2e_sweep,
    )
    result = ablate_loss_e2e.run_ablate_loss(
        train_config_template=_build_train_template(tmp_path),
        eval_config_template=_build_eval_template(tmp_path),
        seeds=[1, 2],
        n_authors=20,
        months=6,
        difficulty="hard",
        pos_weights=[1.0, 2.0],
        focal_gammas=[1.0, 2.0],
        artifact_root=tmp_path / "artifacts" / "experiment_runs",
    )

    summary_path = Path(str(result["summary_path"]))
    csv_path = Path(str(result["csv_path"]))
    assert summary_path.exists()
    assert csv_path.exists()

    frame = pd.read_csv(csv_path)
    # 1 (bce) + 2 (weighted_bce) + 4 (focal)
    assert len(frame) == 7

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "ablate_loss"
    assert isinstance(payload["best_configuration"], dict)
    assert payload["best_configuration"]["loss_label"] in {"bce", "weighted_bce", "focal"}
