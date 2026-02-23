from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from enum import StrEnum
from pathlib import Path

import numpy as np
import typer

from lexical_drift.config import (
    load_eval_e2e_config,
    load_eval_temporal_config,
    load_nn_train_config,
    load_pretrain_contrastive_config,
    load_temporal_train_config,
    load_train_config,
    load_train_e2e_config,
)
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.inference.predict import predict_text
from lexical_drift.training.train_baseline import run_training

app = typer.Typer(help="CLI for lexical drift data generation, training, and inference.")


class Difficulty(StrEnum):
    easy = "easy"
    hard = "hard"


def _dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _resolve_seed_list(
    seeds: str,
    n_seeds: int,
    start_seed: int,
) -> list[int]:
    if seeds:
        parsed = [int(part.strip()) for part in seeds.split(",") if part.strip()]
        if not parsed:
            raise ValueError("No valid seeds provided via --seeds")
        return parsed
    return list(range(start_seed, start_seed + n_seeds))


def _parse_int_list(raw: str, *, name: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"No valid values provided via --{name}")
    return values


def _parse_str_list(raw: str, *, name: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"No valid values provided via --{name}")
    return values


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _format_optional_metric(value: object) -> str:
    if value is None:
        return "na"
    return f"{float(value):.4f}"


def _format_summary_stats(stats: object) -> str:
    if not isinstance(stats, dict):
        return "na"
    required = {"mean", "std", "min", "max"}
    if not required.issubset(stats):
        return "na"
    return (
        f"{float(stats['mean']):.4f}±{float(stats['std']):.4f} "
        f"({float(stats['min']):.4f}..{float(stats['max']):.4f})"
    )


def _print_metric_summary(
    title: str,
    summary: object,
    *,
    prefix: str = "[eval-temporal-sweep]",
) -> None:
    typer.echo(f"{prefix} {title}")
    if not isinstance(summary, dict):
        typer.echo(f"{prefix}   no data")
        return
    for metric in (
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
    ):
        text = _format_summary_stats(summary.get(metric))
        typer.echo(f"{prefix}   {metric:>13} {text}")


def _format_delta(value: object) -> str:
    if value is None:
        return "na"
    return f"{float(value):+.4f}"


def _print_metric_delta(
    title: str,
    delta: object,
    *,
    prefix: str = "[eval-temporal-compare]",
) -> None:
    typer.echo(f"{prefix} {title}")
    if not isinstance(delta, dict):
        typer.echo(f"{prefix}   no data")
        return
    for metric in (
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
    ):
        typer.echo(f"{prefix}   {metric:>13} {_format_delta(delta.get(metric))}")


def _print_metric_significance(
    title: str,
    stats_payload: object,
    *,
    prefix: str = "[eval-temporal-compare]",
) -> None:
    typer.echo(f"{prefix} {title}")
    if not isinstance(stats_payload, dict):
        typer.echo(f"{prefix}   no data")
        return
    for metric in (
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
    ):
        raw = stats_payload.get(metric)
        if not isinstance(raw, dict):
            typer.echo(f"{prefix}   {metric:>13} no paired stats")
            continue
        n_value = int(raw.get("n", 0))
        mean_delta = raw.get("mean_delta")
        ci_low = raw.get("ci_low")
        ci_high = raw.get("ci_high")
        p_value = raw.get("p_value")
        t_stat = raw.get("t_stat")
        if mean_delta is None or ci_low is None or ci_high is None:
            typer.echo(f"{prefix}   {metric:>13} no paired stats")
            continue
        p_text = "na" if p_value is None else f"{float(p_value):.4g}"
        t_text = "na" if t_stat is None else f"{float(t_stat):+.4f}"
        typer.echo(
            f"{prefix}   {metric:>13} "
            f"delta={float(mean_delta):+.4f} "
            f"ci95=[{float(ci_low):+.4f},{float(ci_high):+.4f}] "
            f"t={t_text} p={p_text} n={n_value}"
        )


def _print_drift_correlation_summary(
    title: str,
    correlation_payload: object,
    *,
    prefix: str = "[eval-temporal-compare]",
) -> None:
    typer.echo(f"{prefix} {title}")
    if not isinstance(correlation_payload, dict):
        typer.echo(f"{prefix}   no correlation data")
        return
    summary = correlation_payload.get("summary")
    if not isinstance(summary, dict):
        typer.echo(f"{prefix}   no correlation data")
        return
    for pair_key in (
        "cosine_drift__accuracy_delta_from_ref",
        "l2_drift__accuracy_delta_from_ref",
        "variance_shift__accuracy_delta_from_ref",
        "cosine_drift__f1_delta_from_ref",
        "l2_drift__f1_delta_from_ref",
        "variance_shift__f1_delta_from_ref",
    ):
        stats = summary.get(pair_key)
        if not isinstance(stats, dict):
            typer.echo(f"{prefix}   {pair_key:>36} no data")
            continue
        typer.echo(
            f"{prefix}   {pair_key:>36} "
            f"mean={float(stats['mean']):+.4f} "
            f"std={float(stats['std']):.4f} "
            f"n={int(stats['n'])}"
        )


@app.command("generate-synth")
def generate_synth(
    out: Path = typer.Option(Path("data/raw/synth.csv"), help="Output CSV path."),
    n_authors: int = typer.Option(50, min=1, help="Number of synthetic authors."),
    months: int = typer.Option(12, min=2, help="Number of months per author."),
    seed: int = typer.Option(42, help="Random seed."),
    difficulty: Difficulty = typer.Option(
        Difficulty.easy,
        help="Synthetic generation preset difficulty.",
    ),
    drift_strength: float = typer.Option(
        None,
        help="Override drift strength (default from difficulty preset).",
    ),
    noise_strength: float = typer.Option(
        None,
        help="Override random noise strength (default from difficulty preset).",
    ),
    global_event_strength: float = typer.Option(
        None,
        help="Override global monthly event strength (default from difficulty preset).",
    ),
    topic_shift_strength: float = typer.Option(
        None,
        help="Override topic shift strength (default from difficulty preset).",
    ),
) -> None:
    output = save_synthetic_dataset(
        out_path=out,
        n_authors=n_authors,
        months=months,
        random_seed=seed,
        difficulty=difficulty.value,
        drift_strength=drift_strength,
        noise_strength=noise_strength,
        global_event_strength=global_event_strength,
        topic_shift_strength=topic_shift_strength,
    )
    typer.echo(f"[generate-synth] wrote dataset to {output}")


@app.command("train-baseline")
def train_baseline(
    config: Path = typer.Option(
        Path("configs/train_baseline.yaml"),
        help="Path to training config.",
    ),
) -> None:
    train_config = load_train_config(config)
    result = run_training(train_config)
    typer.echo(f"[train-baseline] accuracy={result['accuracy']:.4f} f1={result['f1']:.4f}")
    typer.echo(f"[train-baseline] model={result['model_path']}")
    typer.echo(f"[train-baseline] metadata={result['metadata_path']}")


@app.command("train-nn")
def train_nn(
    config: Path = typer.Option(
        Path("configs/train_nn.yaml"),
        help="Path to neural training config.",
    ),
) -> None:
    # Lazy import keeps baseline commands usable without the optional torch dependency.
    from lexical_drift.training.train_nn import run_training_nn

    nn_config = load_nn_train_config(config)
    result = run_training_nn(nn_config)
    typer.echo(
        "[train-nn] "
        f"accuracy={result['accuracy']:.4f} "
        f"f1={result['f1']:.4f} "
        f"avg_loss={result['avg_loss']:.4f}"
    )
    typer.echo(f"[train-nn] model={result['model_path']}")
    typer.echo(f"[train-nn] vectorizer={result['vectorizer_path']}")
    typer.echo(f"[train-nn] metadata={result['metadata_path']}")


@app.command("train-temporal")
def train_temporal(
    config: Path = typer.Option(
        Path("configs/train_temporal.yaml"),
        help="Path to temporal training config.",
    ),
) -> None:
    # Lazy import keeps non-temporal commands usable without optional NLP dependencies.
    from lexical_drift.training.train_temporal import run_training_temporal

    temporal_config = load_temporal_train_config(config)
    result = run_training_temporal(temporal_config)
    typer.echo(
        "[train-temporal] "
        f"accuracy={result['accuracy']:.4f} "
        f"f1={result['f1']:.4f} "
        f"avg_loss={result['avg_loss']:.4f}"
    )
    typer.echo(f"[train-temporal] model={result['model_path']}")
    if result["cache_path"]:
        typer.echo(f"[train-temporal] cache={result['cache_path']}")
    typer.echo(f"[train-temporal] metadata={result['metadata_path']}")


@app.command("eval-temporal")
def eval_temporal(
    config: Path = typer.Option(
        Path("configs/eval_temporal.yaml"),
        help="Path to temporal evaluation config.",
    ),
) -> None:
    has_torch = _dependency_available("torch")
    has_transformers = _dependency_available("transformers")
    if not has_torch or not has_transformers:
        typer.echo("[eval-temporal] skipping (torch and/or transformers not installed)")
        return

    # Lazy import keeps baseline commands usable without optional NLP dependencies.
    from lexical_drift.eval.eval_temporal import run_eval_temporal

    eval_config = load_eval_temporal_config(config)
    result = run_eval_temporal(eval_config)
    per_month_summary = dict(result["per_month_summary"])
    confusion = dict(result["final_month_confusion"])
    pred_rates = dict(result["final_month_pred_rates"])
    pred_counts = dict(result["final_month_pred_counts"])
    typer.echo(f"[eval-temporal] model_type={result['model_type']}")
    typer.echo(
        "[eval-temporal] threshold "
        f"mode={result['threshold_mode']} "
        f"metric={result['calibration_metric']} "
        f"chosen={float(result['chosen_threshold']):.4f}"
    )

    typer.echo(
        "[eval-temporal] "
        f"month={result['final_month_index']} "
        f"accuracy={result['final_accuracy']:.4f} "
        f"f1={result['final_f1']:.4f}"
    )
    typer.echo(
        "[eval-temporal] per-month "
        f"accuracy(min/mean/max)="
        f"{per_month_summary['accuracy_min']:.4f}/"
        f"{per_month_summary['accuracy_mean']:.4f}/"
        f"{per_month_summary['accuracy_max']:.4f} "
        f"f1(min/mean/max)="
        f"{per_month_summary['f1_min']:.4f}/"
        f"{per_month_summary['f1_mean']:.4f}/"
        f"{per_month_summary['f1_max']:.4f}"
    )
    typer.echo(
        "[eval-temporal] embedding drift "
        f"cos(min/mean/max)="
        f"{per_month_summary['cosine_drift_min']:.4f}/"
        f"{per_month_summary['cosine_drift_mean']:.4f}/"
        f"{per_month_summary['cosine_drift_max']:.4f} "
        f"l2(min/mean/max)="
        f"{per_month_summary['l2_drift_min']:.4f}/"
        f"{per_month_summary['l2_drift_mean']:.4f}/"
        f"{per_month_summary['l2_drift_max']:.4f} "
        f"var_shift(min/mean/max)="
        f"{per_month_summary['variance_shift_min']:.4f}/"
        f"{per_month_summary['variance_shift_mean']:.4f}/"
        f"{per_month_summary['variance_shift_max']:.4f}"
    )
    threshold_values = np.asarray(
        [float(entry["threshold_used"]) for entry in result["per_month"]],
        dtype=np.float64,
    )
    typer.echo(
        "[eval-temporal] thresholds(min/mean/max)="
        f"{threshold_values.min():.4f}/"
        f"{threshold_values.mean():.4f}/"
        f"{threshold_values.max():.4f}"
    )
    typer.echo(
        "[eval-temporal] month  acc    f1    prec   rec    spec   bal_acc roc_auc pr_auc threshold"
    )
    for entry in result["per_month"]:
        row = dict(entry)
        roc_auc_text = _format_optional_metric(row.get("roc_auc"))
        pr_auc_text = _format_optional_metric(row.get("pr_auc"))
        typer.echo(
            "[eval-temporal] "
            f"{int(row['month_index']):>5d} "
            f"{float(row['accuracy']):>6.4f} "
            f"{float(row['f1']):>6.4f} "
            f"{float(row['precision']):>6.4f} "
            f"{float(row['recall']):>6.4f} "
            f"{float(row['specificity']):>7.4f} "
            f"{float(row['balanced_accuracy']):>8.4f} "
            f"{roc_auc_text:>7} "
            f"{pr_auc_text:>6} "
            f"{float(row['threshold_used']):>9.4f}"
        )
    typer.echo("[eval-temporal] month  true_pos pred_pos tn fp fn tp")
    for entry in result["per_month"]:
        row = dict(entry)
        typer.echo(
            "[eval-temporal] "
            f"{int(row['month_index']):>5d} "
            f"{float(row['true_pos_rate']):>8.4f} "
            f"{float(row['pred_pos_rate']):>8.4f} "
            f"{int(row['tn']):>2d} "
            f"{int(row['fp']):>2d} "
            f"{int(row['fn']):>2d} "
            f"{int(row['tp']):>2d}"
        )
    typer.echo("[eval-temporal] month  cos_drift l2_drift var_shift")
    for entry in result["per_month"]:
        row = dict(entry)
        typer.echo(
            "[eval-temporal] "
            f"{int(row['month_index']):>5d} "
            f"{float(row['cosine_drift']):>9.4f} "
            f"{float(row['l2_drift']):>8.4f} "
            f"{float(row['variance_shift']):>9.4f}"
        )
    typer.echo("[eval-temporal] month  acc_delta f1_delta")
    for entry in result["per_month"]:
        row = dict(entry)
        typer.echo(
            "[eval-temporal] "
            f"{int(row['month_index']):>5d} "
            f"{float(row['accuracy_delta_from_ref']):>9.4f} "
            f"{float(row['f1_delta_from_ref']):>8.4f}"
        )
    typer.echo(
        "[eval-temporal] final confusion "
        f"tp={confusion['tp']} fp={confusion['fp']} "
        f"tn={confusion['tn']} fn={confusion['fn']}"
    )
    typer.echo(
        "[eval-temporal] final prediction rates "
        f"pred_0={pred_rates['pred_0_rate']:.4f} "
        f"pred_1={pred_rates['pred_1_rate']:.4f} "
        f"(counts: pred_0={pred_counts['pred_0']}, pred_1={pred_counts['pred_1']})"
    )
    typer.echo(f"[eval-temporal] metrics={result['metrics_path']}")
    typer.echo(f"[eval-temporal] per-month-csv={result['per_month_csv_path']}")
    typer.echo(f"[eval-temporal] model={result['model_path']}")
    if result["cache_path"]:
        typer.echo(f"[eval-temporal] cache={result['cache_path']}")


@app.command("eval-temporal-real")
def eval_temporal_real(
    dataset: str = typer.Option(
        "sample_local",
        help="Real dataset loader name (currently: sample_local).",
    ),
    path: Path = typer.Option(
        Path("data/raw/real_sample.csv"),
        help="Path to local real dataset CSV.",
    ),
    config: Path = typer.Option(
        Path("configs/eval_temporal.yaml"),
        help="Path to temporal evaluation config.",
    ),
) -> None:
    has_transformers = _dependency_available("transformers")
    if not has_transformers:
        typer.echo("[eval-temporal-real] skipping (transformers not installed)")
        return

    from lexical_drift.datasets.real import load_real_dataset
    from lexical_drift.eval.eval_temporal import run_eval_temporal

    eval_config = load_eval_temporal_config(config)
    if eval_config.model_type in {"gru", "attention", "transformer"} and not _dependency_available(
        "torch"
    ):
        typer.echo("[eval-temporal-real] skipping (torch not installed for temporal model_type)")
        return

    normalized = load_real_dataset(name=dataset, path=path)
    output_dir = Path(eval_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "real_dataset_prepared.csv"
    normalized.to_csv(normalized_path, index=False)
    typer.echo(f"[eval-temporal-real] prepared dataset={normalized_path}")

    run_config = replace(eval_config, input_path=str(normalized_path))
    result = run_eval_temporal(run_config)
    typer.echo(
        "[eval-temporal-real] "
        f"month={result['final_month_index']} "
        f"accuracy={result['final_accuracy']:.4f} "
        f"f1={result['final_f1']:.4f}"
    )
    typer.echo(f"[eval-temporal-real] metrics={result['metrics_path']}")
    typer.echo(f"[eval-temporal-real] model={result['model_path']}")
    if result["cache_path"]:
        typer.echo(f"[eval-temporal-real] cache={result['cache_path']}")


@app.command("eval-temporal-sweep")
def eval_temporal_sweep(
    config: Path = typer.Option(
        Path("configs/eval_temporal.yaml"),
        help="Path to temporal evaluation config template.",
    ),
    seeds: str = typer.Option(
        "",
        help="Comma-separated seeds (e.g. 1,2,3). Overrides --n-seeds/--start-seed.",
    ),
    n_seeds: int = typer.Option(10, min=1, help="Number of sequential seeds to run."),
    start_seed: int = typer.Option(1, help="Starting seed when --seeds is not provided."),
    n_authors: int = typer.Option(50, min=1, help="Synthetic authors per seed."),
    months: int = typer.Option(12, min=2, help="Synthetic months per author."),
    difficulty: Difficulty = typer.Option(
        Difficulty.easy,
        help="Synthetic generation preset difficulty.",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts"),
        help="Root directory for sweep outputs.",
    ),
    results_path: str = typer.Option(
        "",
        help="JSONL output file path (default: <artifact_root>/eval_temporal_sweep.jsonl).",
    ),
) -> None:
    has_torch = _dependency_available("torch")
    has_transformers = _dependency_available("transformers")
    if not has_torch or not has_transformers:
        typer.echo("[eval-temporal-sweep] skipping (torch and/or transformers not installed)")
        return

    from lexical_drift.eval.eval_temporal_sweep import run_eval_temporal_sweep

    eval_template = load_eval_temporal_config(config)
    seed_list = _resolve_seed_list(seeds, n_seeds, start_seed)
    output_results = (
        Path(results_path) if results_path else artifact_root / "eval_temporal_sweep.jsonl"
    )

    result = run_eval_temporal_sweep(
        config_template=eval_template,
        seeds=seed_list,
        n_authors=n_authors,
        months=months,
        difficulty=difficulty.value,
        artifact_root=artifact_root,
        results_path=output_results,
    )

    typer.echo(f"[eval-temporal-sweep] results={result['results_path']}")
    typer.echo(f"[eval-temporal-sweep] records-csv={result['sweep_records_csv_path']}")
    typer.echo(f"[eval-temporal-sweep] model_type={result['model_type']}")
    typer.echo(
        "[eval-temporal-sweep] "
        f"runs={result['total_runs']} "
        f"success={result['success_count']} "
        f"failed={result['failure_count']}"
    )
    if result["failed_seeds"]:
        typer.echo(f"[eval-temporal-sweep] failed_seeds={result['failed_seeds']}")

    final_month_index = result["final_month_index"]
    month_label = "mixed" if final_month_index is None else str(final_month_index)
    _print_metric_summary(
        f"FINAL MONTH (month={month_label}): metric mean±std (min..max)",
        result["final_month_summary"],
    )
    _print_metric_summary(
        "ALL EVAL MONTHS: metric mean±std (min..max)",
        result["all_eval_months_summary"],
    )


@app.command("eval-temporal-compare")
def eval_temporal_compare(
    config_a: Path = typer.Option(
        Path("configs/eval_temporal.yaml"),
        help="Path to eval-temporal config A.",
    ),
    config_b: Path = typer.Option(
        Path("configs/eval_temporal.yaml"),
        help="Path to eval-temporal config B.",
    ),
    seeds: str = typer.Option(
        "",
        help="Comma-separated seeds (e.g. 1,2,3). Overrides --n-seeds/--start-seed.",
    ),
    n_seeds: int = typer.Option(10, min=1, help="Number of sequential seeds to run."),
    start_seed: int = typer.Option(1, help="Starting seed when --seeds is not provided."),
    n_authors: int = typer.Option(200, min=1, help="Synthetic authors per seed."),
    months: int = typer.Option(12, min=2, help="Synthetic months per author."),
    difficulty: Difficulty = typer.Option(
        Difficulty.hard,
        help="Synthetic generation preset difficulty.",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts"),
        help="Root directory for compare outputs.",
    ),
) -> None:
    has_torch = _dependency_available("torch")
    has_transformers = _dependency_available("transformers")
    if not has_torch or not has_transformers:
        typer.echo("[eval-temporal-compare] skipping (torch and/or transformers not installed)")
        return

    from lexical_drift.eval.eval_temporal_compare import run_eval_temporal_compare

    cfg_a = load_eval_temporal_config(config_a)
    cfg_b = load_eval_temporal_config(config_b)
    seed_list = _resolve_seed_list(seeds, n_seeds, start_seed)

    result = run_eval_temporal_compare(
        config_a_template=cfg_a,
        config_b_template=cfg_b,
        config_a_path=config_a,
        config_b_path=config_b,
        seeds=seed_list,
        n_authors=n_authors,
        months=months,
        difficulty=difficulty.value,
        artifact_root=artifact_root,
    )

    summary_a = dict(result["summary_a"])
    summary_b = dict(result["summary_b"])
    typer.echo(f"[eval-temporal-compare] summary={result['summary_path']}")
    typer.echo(f"[eval-temporal-compare] seed-deltas-csv={result['compare_seed_deltas_csv_path']}")
    typer.echo(
        "[eval-temporal-compare] "
        f"A model_type={summary_a['model_type']} "
        f"success={summary_a['success_count']}/{summary_a['total_runs']} "
        f"failed={summary_a['failure_count']}"
    )
    typer.echo(
        "[eval-temporal-compare] "
        f"B model_type={summary_b['model_type']} "
        f"success={summary_b['success_count']}/{summary_b['total_runs']} "
        f"failed={summary_b['failure_count']}"
    )

    month_a = summary_a["final_month_index"]
    month_b = summary_b["final_month_index"]
    month_label_a = "mixed" if month_a is None else str(month_a)
    month_label_b = "mixed" if month_b is None else str(month_b)
    _print_metric_summary(
        f"A FINAL MONTH (month={month_label_a}): metric mean±std (min..max)",
        result["final_month_summary_a"],
        prefix="[eval-temporal-compare]",
    )
    _print_metric_summary(
        f"B FINAL MONTH (month={month_label_b}): metric mean±std (min..max)",
        result["final_month_summary_b"],
        prefix="[eval-temporal-compare]",
    )
    _print_metric_delta(
        "DELTA FINAL MONTH (B_mean - A_mean)",
        result["final_month_delta"],
        prefix="[eval-temporal-compare]",
    )
    _print_metric_significance(
        "SIGNIFICANCE FINAL MONTH (paired B-A)",
        result["final_month_stats"],
        prefix="[eval-temporal-compare]",
    )

    _print_metric_summary(
        "A ALL EVAL MONTHS: metric mean±std (min..max)",
        result["all_months_summary_a"],
        prefix="[eval-temporal-compare]",
    )
    _print_metric_summary(
        "B ALL EVAL MONTHS: metric mean±std (min..max)",
        result["all_months_summary_b"],
        prefix="[eval-temporal-compare]",
    )
    _print_metric_delta(
        "DELTA ALL EVAL MONTHS (B_mean - A_mean)",
        result["all_months_delta"],
        prefix="[eval-temporal-compare]",
    )
    _print_metric_significance(
        "SIGNIFICANCE ALL EVAL MONTHS (paired B-A)",
        result["all_months_stats"],
        prefix="[eval-temporal-compare]",
    )
    _print_drift_correlation_summary(
        "A DRIFT/PERF CORRELATION (per-seed summary)",
        result["drift_performance_correlation_a"],
        prefix="[eval-temporal-compare]",
    )
    _print_drift_correlation_summary(
        "B DRIFT/PERF CORRELATION (per-seed summary)",
        result["drift_performance_correlation_b"],
        prefix="[eval-temporal-compare]",
    )


@app.command("ablation-train-months")
def ablation_train_months(
    config: Path = typer.Option(
        Path("configs/eval_temporal.yaml"),
        help="Path to temporal evaluation config template.",
    ),
    train_months: str = typer.Option(
        "3,6,9",
        help="Comma-separated train_months values (e.g. 3,6,9).",
    ),
    seeds: str = typer.Option(
        "",
        help="Comma-separated seeds (e.g. 1,2,3). Overrides --n-seeds/--start-seed.",
    ),
    n_seeds: int = typer.Option(3, min=1, help="Number of sequential seeds to run."),
    start_seed: int = typer.Option(1, help="Starting seed when --seeds is not provided."),
    n_authors: int = typer.Option(50, min=1, help="Synthetic authors per seed."),
    months: int = typer.Option(12, min=2, help="Synthetic months per author."),
    difficulty: Difficulty = typer.Option(
        Difficulty.hard,
        help="Synthetic generation preset difficulty.",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts"),
        help="Root directory for ablation outputs.",
    ),
) -> None:
    has_torch = _dependency_available("torch")
    has_transformers = _dependency_available("transformers")
    if not has_torch or not has_transformers:
        typer.echo("[ablation-train-months] skipping (torch and/or transformers not installed)")
        return

    from lexical_drift.eval.ablation_train_months import run_ablation_train_months

    eval_template = load_eval_temporal_config(config)
    seed_list = _resolve_seed_list(seeds, n_seeds, start_seed)
    train_months_values = _parse_int_list(train_months, name="train-months")
    result = run_ablation_train_months(
        config_template=eval_template,
        train_months_values=train_months_values,
        seeds=seed_list,
        n_authors=n_authors,
        months=months,
        difficulty=difficulty.value,
        artifact_root=artifact_root,
    )

    typer.echo(f"[ablation-train-months] summary={result['summary_path']}")
    typer.echo(f"[ablation-train-months] plot={result['plot_path']}")
    for row in result["rows"]:
        row_data = dict(row)
        typer.echo(
            "[ablation-train-months] "
            f"train_months={int(row_data['train_months'])} "
            f"final_accuracy_mean={_format_optional_metric(row_data['final_accuracy_mean'])} "
            f"final_f1_mean={_format_optional_metric(row_data['final_f1_mean'])} "
            f"final_bal_acc_mean={_format_optional_metric(row_data['final_balanced_accuracy_mean'])}"
        )


@app.command("ablation-encoder")
def ablation_encoder(
    config: Path = typer.Option(
        Path("configs/eval_temporal.yaml"),
        help="Path to temporal evaluation config template.",
    ),
    encoders: str = typer.Option(
        "distilbert-base-uncased,bert-base-uncased",
        help="Comma-separated encoder models.",
    ),
    seeds: str = typer.Option(
        "",
        help="Comma-separated seeds (e.g. 1,2,3). Overrides --n-seeds/--start-seed.",
    ),
    n_seeds: int = typer.Option(3, min=1, help="Number of sequential seeds to run."),
    start_seed: int = typer.Option(1, help="Starting seed when --seeds is not provided."),
    n_authors: int = typer.Option(50, min=1, help="Synthetic authors per seed."),
    months: int = typer.Option(12, min=2, help="Synthetic months per author."),
    difficulty: Difficulty = typer.Option(
        Difficulty.hard,
        help="Synthetic generation preset difficulty.",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts"),
        help="Root directory for ablation outputs.",
    ),
) -> None:
    has_torch = _dependency_available("torch")
    has_transformers = _dependency_available("transformers")
    if not has_torch or not has_transformers:
        typer.echo("[ablation-encoder] skipping (torch and/or transformers not installed)")
        return

    from lexical_drift.eval.ablation_encoder import run_ablation_encoder

    eval_template = load_eval_temporal_config(config)
    seed_list = _resolve_seed_list(seeds, n_seeds, start_seed)
    encoder_models = _parse_str_list(encoders, name="encoders")
    result = run_ablation_encoder(
        config_template=eval_template,
        encoder_models=encoder_models,
        seeds=seed_list,
        n_authors=n_authors,
        months=months,
        difficulty=difficulty.value,
        artifact_root=artifact_root,
    )

    typer.echo(f"[ablation-encoder] summary={result['summary_path']}")
    typer.echo(f"[ablation-encoder] plot={result['plot_path']}")
    for row in result["rows"]:
        row_data = dict(row)
        typer.echo(
            "[ablation-encoder] "
            f"encoder={row_data['encoder_model']} "
            f"final_accuracy_mean={_format_optional_metric(row_data['final_accuracy_mean'])} "
            f"final_f1_mean={_format_optional_metric(row_data['final_f1_mean'])} "
            f"final_bal_acc_mean={_format_optional_metric(row_data['final_balanced_accuracy_mean'])}"
        )


@app.command("render-report")
def render_report(
    compare_summary: Path = typer.Option(
        Path("artifacts/eval_temporal_compare_summary.json"),
        help="Path to eval-temporal-compare summary JSON.",
    ),
    out: Path = typer.Option(
        Path("docs/report.md"),
        help="Output markdown report path.",
    ),
) -> None:
    from lexical_drift.eval.report import render_compare_report

    output_path = render_compare_report(
        compare_summary_path=compare_summary,
        out_path=out,
    )
    typer.echo(f"[render-report] wrote report to {output_path}")


@app.command("train-e2e")
def train_e2e(
    config: Path = typer.Option(
        Path("configs/train_e2e_temporal.yaml"),
        help="Path to end-to-end temporal training config.",
    ),
) -> None:
    has_torch = _dependency_available("torch")
    has_transformers = _dependency_available("transformers")
    if not has_torch or not has_transformers:
        typer.echo("[train-e2e] skipping (torch and/or transformers not installed)")
        return

    from lexical_drift.train.e2e_temporal import run_train_e2e

    train_config = load_train_e2e_config(config)
    result = run_train_e2e(train_config)
    typer.echo(
        "[train-e2e] "
        f"month={result['final_month_index']} "
        f"accuracy={result['final_accuracy']:.4f} "
        f"f1={result['final_f1']:.4f}"
    )
    typer.echo(f"[train-e2e] output_dir={result['output_dir']}")
    typer.echo(f"[train-e2e] model={result['model_path']}")
    typer.echo(f"[train-e2e] metrics={result['metrics_path']}")
    typer.echo(f"[train-e2e] per-month-csv={result['per_month_csv_path']}")
    typer.echo(f"[train-e2e] metadata={result['run_metadata_path']}")


@app.command("eval-e2e")
def eval_e2e(
    config: Path = typer.Option(
        Path("configs/eval_e2e_temporal.yaml"),
        help="Path to end-to-end temporal evaluation config.",
    ),
) -> None:
    has_torch = _dependency_available("torch")
    has_transformers = _dependency_available("transformers")
    if not has_torch or not has_transformers:
        typer.echo("[eval-e2e] skipping (torch and/or transformers not installed)")
        return

    from lexical_drift.train.e2e_temporal import run_eval_e2e

    eval_config = load_eval_e2e_config(config)
    result = run_eval_e2e(eval_config)
    typer.echo(
        "[eval-e2e] "
        f"month={result['final_month_index']} "
        f"accuracy={result['final_accuracy']:.4f} "
        f"f1={result['final_f1']:.4f}"
    )
    typer.echo(f"[eval-e2e] output_dir={result['output_dir']}")
    typer.echo(f"[eval-e2e] model={result['model_path']}")
    typer.echo(f"[eval-e2e] metrics={result['metrics_path']}")
    typer.echo(f"[eval-e2e] per-month-csv={result['per_month_csv_path']}")
    typer.echo(f"[eval-e2e] metadata={result['run_metadata_path']}")


@app.command("pretrain-contrastive")
def pretrain_contrastive(
    config: Path = typer.Option(
        Path("configs/pretrain_contrastive.yaml"),
        help="Path to temporal contrastive pretraining config.",
    ),
) -> None:
    has_torch = _dependency_available("torch")
    has_transformers = _dependency_available("transformers")
    if not has_torch or not has_transformers:
        typer.echo("[pretrain-contrastive] skipping (torch and/or transformers not installed)")
        return

    from lexical_drift.train.contrastive_temporal import run_pretrain_contrastive

    pretrain_config = load_pretrain_contrastive_config(config)
    result = run_pretrain_contrastive(pretrain_config)
    typer.echo(
        f"[pretrain-contrastive] pairs={result['n_pairs']} final_loss={result['final_loss']:.4f}"
    )
    typer.echo(f"[pretrain-contrastive] output_dir={result['output_dir']}")
    typer.echo(f"[pretrain-contrastive] checkpoint={result['checkpoint_path']}")
    typer.echo(f"[pretrain-contrastive] metrics={result['metrics_path']}")
    typer.echo(f"[pretrain-contrastive] metadata={result['run_metadata_path']}")


@app.command("benchmark")
def benchmark(
    seeds: str = typer.Option(
        "",
        help="Comma-separated seeds (e.g. 1,2,3). Overrides --n-seeds/--start-seed.",
    ),
    n_seeds: int = typer.Option(10, min=1, help="Number of sequential seeds to run."),
    start_seed: int = typer.Option(1, help="Starting seed when --seeds is not provided."),
    n_authors: int = typer.Option(50, min=1, help="Synthetic authors per seed."),
    months: int = typer.Option(12, min=2, help="Synthetic months per author."),
    artifact_root: Path = typer.Option(
        Path("artifacts"),
        help="Root directory for benchmark outputs.",
    ),
    results_path: str = typer.Option(
        "",
        help="JSONL output file path (default: <artifact_root>/benchmark_results.jsonl).",
    ),
    overwrite_default_synth: bool = typer.Option(
        False,
        help=(
            "Overwrite data/raw/synth.csv for each seed instead of using temp benchmark data files."
        ),
    ),
    baseline_config: Path = typer.Option(
        Path("configs/train_baseline.yaml"),
        help="Path to baseline config template.",
    ),
    nn_config: Path = typer.Option(
        Path("configs/train_nn.yaml"),
        help="Path to NN config template.",
    ),
    temporal_config: Path = typer.Option(
        Path("configs/train_temporal.yaml"),
        help="Path to temporal config template.",
    ),
) -> None:
    seed_list = _resolve_seed_list(seeds, n_seeds, start_seed)
    output_root = artifact_root
    output_root.mkdir(parents=True, exist_ok=True)
    output_results = (
        Path(results_path) if results_path else (output_root / "benchmark_results.jsonl")
    )

    baseline_template = load_train_config(baseline_config)
    nn_template = load_nn_train_config(nn_config)
    temporal_template = load_temporal_train_config(temporal_config)

    has_torch = _dependency_available("torch")
    has_transformers = _dependency_available("transformers")

    run_nn_fn = None
    if has_torch:
        from lexical_drift.training.train_nn import run_training_nn

        run_nn_fn = run_training_nn
    else:
        typer.echo("[benchmark] skipping train-nn (torch is not installed)")

    run_temporal_fn = None
    if has_torch and has_transformers:
        from lexical_drift.training.train_temporal import run_training_temporal

        run_temporal_fn = run_training_temporal
    else:
        typer.echo("[benchmark] skipping train-temporal (torch and/or transformers missing)")

    all_records: list[dict[str, object]] = []

    for seed in seed_list:
        seed_root = output_root / "benchmark_runs" / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)

        if overwrite_default_synth:
            synth_path = Path("data/raw/synth.csv")
        else:
            synth_path = output_root / "benchmark_data" / f"synth_seed_{seed}.csv"

        save_synthetic_dataset(
            out_path=synth_path,
            n_authors=n_authors,
            months=months,
            random_seed=seed,
        )
        typer.echo(f"[benchmark] seed={seed} synth={synth_path}")

        baseline_cfg = replace(
            baseline_template,
            input_path=str(synth_path),
            output_dir=str(seed_root / "baseline"),
            random_seed=seed,
        )
        baseline_result = run_training(baseline_cfg)
        baseline_record = {
            "seed": seed,
            "model": "baseline",
            "status": "ok",
            "accuracy": baseline_result["accuracy"],
            "f1": baseline_result["f1"],
            "model_path": baseline_result["model_path"],
            "metadata_path": baseline_result["metadata_path"],
        }
        _append_jsonl(output_results, baseline_record)
        all_records.append(baseline_record)

        if run_nn_fn is None:
            nn_record = {
                "seed": seed,
                "model": "nn",
                "status": "skipped",
                "reason": "torch not installed",
            }
        else:
            nn_cfg = replace(
                nn_template,
                input_path=str(synth_path),
                output_dir=str(seed_root / "nn"),
                random_seed=seed,
            )
            nn_result = run_nn_fn(nn_cfg)
            nn_record = {
                "seed": seed,
                "model": "nn",
                "status": "ok",
                "accuracy": nn_result["accuracy"],
                "f1": nn_result["f1"],
                "avg_loss": nn_result["avg_loss"],
                "model_path": nn_result["model_path"],
                "metadata_path": nn_result["metadata_path"],
            }
        _append_jsonl(output_results, nn_record)
        all_records.append(nn_record)

        if run_temporal_fn is None:
            temporal_record = {
                "seed": seed,
                "model": "temporal",
                "status": "skipped",
                "reason": "torch and/or transformers not installed",
            }
        else:
            temporal_cfg = replace(
                temporal_template,
                input_path=str(synth_path),
                output_dir=str(seed_root / "temporal"),
                cache_dir=str(seed_root / "temporal_cache"),
                random_seed=seed,
            )
            temporal_result = run_temporal_fn(temporal_cfg)
            temporal_record = {
                "seed": seed,
                "model": "temporal",
                "status": "ok",
                "accuracy": temporal_result["accuracy"],
                "f1": temporal_result["f1"],
                "avg_loss": temporal_result["avg_loss"],
                "model_path": temporal_result["model_path"],
                "metadata_path": temporal_result["metadata_path"],
            }
        _append_jsonl(output_results, temporal_record)
        all_records.append(temporal_record)

    typer.echo(f"[benchmark] wrote results to {output_results}")

    for model_name in ("baseline", "nn", "temporal"):
        model_records = [
            record
            for record in all_records
            if record["model"] == model_name and record["status"] == "ok"
        ]
        if not model_records:
            typer.echo(f"[benchmark] {model_name}: no successful runs")
            continue

        accuracy_values = np.asarray([float(record["accuracy"]) for record in model_records])
        f1_values = np.asarray([float(record["f1"]) for record in model_records])
        typer.echo(
            f"[benchmark] {model_name} "
            f"accuracy mean={accuracy_values.mean():.4f} std={accuracy_values.std():.4f} "
            f"f1 mean={f1_values.mean():.4f} std={f1_values.std():.4f}"
        )


@app.command("predict")
def predict(
    model: Path = typer.Option(
        Path("artifacts/baseline.joblib"),
        help="Path to trained model artifact.",
    ),
    text: str = typer.Option(..., help="Input text to score for drift."),
) -> None:
    result = predict_text(model, text)
    typer.echo(
        f"[predict] drift_label={result['drift_label']} drift_score={result['drift_score']:.4f}"
    )


if __name__ == "__main__":
    app()
