from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class TrainConfig:
    input_path: str
    output_dir: str
    test_size: float
    random_seed: int
    max_features: int
    C: float


def load_train_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    required = {"input_path", "output_dir", "test_size", "random_seed", "max_features", "C"}
    missing = sorted(required - set(raw.keys()))
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Missing config keys: {missing_keys}")

    config = TrainConfig(
        input_path=str(raw["input_path"]),
        output_dir=str(raw["output_dir"]),
        test_size=float(raw["test_size"]),
        random_seed=int(raw["random_seed"]),
        max_features=int(raw["max_features"]),
        C=float(raw["C"]),
    )

    if not 0.0 < config.test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if config.max_features <= 0:
        raise ValueError("max_features must be > 0")
    if config.C <= 0:
        raise ValueError("C must be > 0")

    return config
