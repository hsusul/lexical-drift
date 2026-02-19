from __future__ import annotations

import pandas as pd

from lexical_drift.config import TrainConfig
from lexical_drift.datasets.synthetic import generate_synthetic_dataset, save_synthetic_dataset
from lexical_drift.training.train_baseline import run_training


def test_synth_has_both_classes_and_ordered_months() -> None:
    months = 6
    frame = generate_synthetic_dataset(
        n_authors=30,
        months=months,
        random_seed=11,
    )

    assert set(frame["drift_label"].unique().tolist()) == {0, 1}

    for _author_id, author_df in frame.groupby("author_id"):
        month_values = author_df["month_index"].astype(int).tolist()
        assert month_values == list(range(months))


def test_hard_mode_not_trivially_separable_for_baseline(tmp_path) -> None:
    data_path = tmp_path / "hard_synth.csv"
    output_dir = tmp_path / "artifacts"

    save_synthetic_dataset(
        data_path,
        n_authors=80,
        months=10,
        random_seed=123,
        difficulty="hard",
    )

    frame = pd.read_csv(data_path)
    assert set(frame["drift_label"].unique().tolist()) == {0, 1}

    config = TrainConfig(
        input_path=str(data_path),
        output_dir=str(output_dir),
        test_size=0.25,
        random_seed=123,
        max_features=2000,
        C=1.0,
    )
    result = run_training(config)

    assert float(result["accuracy"]) < 0.95
