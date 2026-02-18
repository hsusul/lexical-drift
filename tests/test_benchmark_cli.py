from __future__ import annotations

import json

from typer.testing import CliRunner

from lexical_drift import cli

runner = CliRunner()


def test_benchmark_runs_baseline_and_skips_optional_models(tmp_path, monkeypatch) -> None:
    artifact_root = tmp_path / "artifacts"

    def fake_dependency_available(module_name: str) -> bool:
        if module_name in {"torch", "transformers"}:
            return False
        return True

    monkeypatch.setattr(cli, "_dependency_available", fake_dependency_available)

    result = runner.invoke(
        cli.app,
        [
            "benchmark",
            "--seeds",
            "1,2",
            "--n-authors",
            "6",
            "--months",
            "4",
            "--artifact-root",
            str(artifact_root),
        ],
    )
    assert result.exit_code == 0

    results_path = artifact_root / "benchmark_results.jsonl"
    assert results_path.exists()

    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 6

    baseline_records = [record for record in records if record["model"] == "baseline"]
    nn_records = [record for record in records if record["model"] == "nn"]
    temporal_records = [record for record in records if record["model"] == "temporal"]

    assert len(baseline_records) == 2
    assert all(record["status"] == "ok" for record in baseline_records)
    assert all("accuracy" in record and "f1" in record for record in baseline_records)

    assert len(nn_records) == 2
    assert all(record["status"] == "skipped" for record in nn_records)

    assert len(temporal_records) == 2
    assert all(record["status"] == "skipped" for record in temporal_records)
