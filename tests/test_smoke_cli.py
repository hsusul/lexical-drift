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
