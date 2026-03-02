from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from lexical_drift.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "generate-synth" in result.output
    assert "train-baseline" in result.output
    assert "train-nn" in result.output
    assert "train-temporal" in result.output
    assert "train-e2e" in result.output
    assert "eval-e2e" in result.output
    assert "eval-e2e-sweep" in result.output
    assert "ablate-time-embeddings" in result.output
    assert "ablate-loss" in result.output
    assert "summarize-experiments" in result.output
    assert "pretrain-contrastive" in result.output
    assert "pretrain-temporal-order" in result.output
    assert "train-multitask" in result.output
    assert "ablation-drift-weight" in result.output
    assert "ablation-time-embeddings" in result.output
    assert "eval-temporal" in result.output
    assert "eval-temporal-real" in result.output
    assert "eval-temporal-sweep" in result.output
    assert "eval-temporal-compare" in result.output
    assert "prepare-real" in result.output
    assert "render-report" in result.output
    assert "benchmark" in result.output
    assert "predict" in result.output


def test_generate_synth_writes_expected_columns(tmp_path) -> None:
    out_csv = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "generate-synth",
            "--out",
            str(out_csv),
            "--n-authors",
            "6",
            "--months",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert out_csv.exists()

    frame = pd.read_csv(out_csv)
    assert set(frame.columns) == {"author_id", "month_index", "text", "drift_label"}
    assert len(frame) == 24
