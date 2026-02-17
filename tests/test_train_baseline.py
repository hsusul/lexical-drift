from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from lexical_drift.cli import app
from lexical_drift.datasets.synthetic import save_synthetic_dataset


runner = CliRunner()


def test_train_baseline_creates_artifacts(tmp_path) -> None:
    data_path = tmp_path / "tiny_synth.csv"
    output_dir = tmp_path / "artifacts"
    config_path = tmp_path / "train.yaml"

    save_synthetic_dataset(data_path, n_authors=10, months=6, random_seed=7)

    config = {
        "input_path": str(data_path),
        "output_dir": str(output_dir),
        "test_size": 0.25,
        "random_seed": 7,
        "max_features": 400,
        "C": 1.0,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = runner.invoke(app, ["train-baseline", "--config", str(config_path)])
    assert result.exit_code == 0

    model_path = output_dir / "baseline.joblib"
    metadata_path = output_dir / "metadata.json"
    assert model_path.exists()
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "metrics" in metadata
    assert "accuracy" in metadata["metrics"]
    assert "f1" in metadata["metrics"]
