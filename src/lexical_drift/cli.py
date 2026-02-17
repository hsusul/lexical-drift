from __future__ import annotations

from pathlib import Path

import typer

from lexical_drift.config import load_train_config
from lexical_drift.datasets.synthetic import save_synthetic_dataset
from lexical_drift.inference.predict import predict_text
from lexical_drift.training.train_baseline import run_training

app = typer.Typer(help="CLI for lexical drift data generation, training, and inference.")


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
