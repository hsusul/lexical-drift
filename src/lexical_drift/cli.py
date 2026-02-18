from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import typer

from lexical_drift.config import (
    load_nn_train_config,
    load_temporal_train_config,
    load_train_config,
)
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.inference.predict import predict_text
from lexical_drift.training.train_baseline import run_training

app = typer.Typer(help="CLI for lexical drift data generation, training, and inference.")


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


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.command("generate-synth")
def generate_synth(
    out: Path = typer.Option(Path("data/raw/synth.csv"), help="Output CSV path."),
    n_authors: int = typer.Option(50, min=1, help="Number of synthetic authors."),
    months: int = typer.Option(12, min=2, help="Number of months per author."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    output = save_synthetic_dataset(
        out_path=out,
        n_authors=n_authors,
        months=months,
        random_seed=seed,
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
            "Overwrite data/raw/synth.csv for each seed instead of using temp benchmark "
            "data files."
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
