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


@dataclass(slots=True)
class NNTrainConfig:
    input_path: str
    output_dir: str
    test_size: float
    random_seed: int
    max_features: int
    lr: float
    batch_size: int
    epochs: int
    hidden_dim: int
    dropout: float


@dataclass(slots=True)
class TemporalTrainConfig:
    input_path: str
    output_dir: str
    test_size: float
    random_seed: int
    max_features: int
    encoder_model: str
    max_length: int
    batch_size: int
    cache_embeddings: bool
    cache_dir: str
    gru_hidden_dim: int
    gru_layers: int
    dropout: float
    lr: float
    epochs: int


@dataclass(slots=True)
class EvalTemporalConfig:
    input_path: str
    output_dir: str
    random_seed: int
    encoder_model: str
    max_length: int
    batch_size: int
    cache_embeddings: bool
    cache_dir: str
    train_months: int
    gru_hidden_dim: int
    gru_layers: int
    dropout: float
    lr: float
    epochs: int
    model_type: str = "gru"
    threshold_mode: str = "fixed"
    fixed_threshold: float = 0.5
    calibration_metric: str = "balanced_accuracy"
    test_size: float | None = None


@dataclass(slots=True)
class TrainE2EConfig:
    input_path: str
    output_dir: str
    random_seed: int
    encoder_model: str
    max_length: int
    batch_size: int
    train_months: int
    gru_hidden_dim: int
    gru_layers: int
    dropout: float
    lr: float
    epochs: int
    test_size: float
    pooling: str = "cls"
    freeze_encoder: bool = False
    pretrained_encoder_path: str = ""


@dataclass(slots=True)
class EvalE2EConfig:
    input_path: str
    output_dir: str
    random_seed: int
    encoder_model: str
    max_length: int
    batch_size: int
    train_months: int
    checkpoint_path: str
    test_size: float
    pooling: str = "cls"
    threshold: float = 0.5


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


def load_nn_train_config(path: str | Path) -> NNTrainConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    required = {
        "input_path",
        "output_dir",
        "test_size",
        "random_seed",
        "max_features",
        "lr",
        "batch_size",
        "epochs",
        "hidden_dim",
        "dropout",
    }
    missing = sorted(required - set(raw.keys()))
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Missing config keys: {missing_keys}")

    config = NNTrainConfig(
        input_path=str(raw["input_path"]),
        output_dir=str(raw["output_dir"]),
        test_size=float(raw["test_size"]),
        random_seed=int(raw["random_seed"]),
        max_features=int(raw["max_features"]),
        lr=float(raw["lr"]),
        batch_size=int(raw["batch_size"]),
        epochs=int(raw["epochs"]),
        hidden_dim=int(raw["hidden_dim"]),
        dropout=float(raw["dropout"]),
    )

    if not 0.0 < config.test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if config.max_features <= 0:
        raise ValueError("max_features must be > 0")
    if config.lr <= 0:
        raise ValueError("lr must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be > 0")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")

    return config


def load_temporal_train_config(path: str | Path) -> TemporalTrainConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    required = {
        "input_path",
        "output_dir",
        "test_size",
        "random_seed",
        "max_features",
        "encoder_model",
        "max_length",
        "batch_size",
        "cache_embeddings",
        "cache_dir",
        "gru_hidden_dim",
        "gru_layers",
        "dropout",
        "lr",
        "epochs",
    }
    missing = sorted(required - set(raw.keys()))
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Missing config keys: {missing_keys}")

    config = TemporalTrainConfig(
        input_path=str(raw["input_path"]),
        output_dir=str(raw["output_dir"]),
        test_size=float(raw["test_size"]),
        random_seed=int(raw["random_seed"]),
        max_features=int(raw["max_features"]),
        encoder_model=str(raw["encoder_model"]),
        max_length=int(raw["max_length"]),
        batch_size=int(raw["batch_size"]),
        cache_embeddings=bool(raw["cache_embeddings"]),
        cache_dir=str(raw["cache_dir"]),
        gru_hidden_dim=int(raw["gru_hidden_dim"]),
        gru_layers=int(raw["gru_layers"]),
        dropout=float(raw["dropout"]),
        lr=float(raw["lr"]),
        epochs=int(raw["epochs"]),
    )

    if not 0.0 < config.test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if config.max_features <= 0:
        raise ValueError("max_features must be > 0")
    if not config.encoder_model.strip():
        raise ValueError("encoder_model must be non-empty")
    if config.max_length <= 0:
        raise ValueError("max_length must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.gru_hidden_dim <= 0:
        raise ValueError("gru_hidden_dim must be > 0")
    if config.gru_layers <= 0:
        raise ValueError("gru_layers must be > 0")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if config.lr <= 0:
        raise ValueError("lr must be > 0")
    if config.epochs <= 0:
        raise ValueError("epochs must be > 0")

    return config


def load_eval_temporal_config(path: str | Path) -> EvalTemporalConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    required = {
        "input_path",
        "output_dir",
        "random_seed",
        "encoder_model",
        "max_length",
        "batch_size",
        "cache_embeddings",
        "cache_dir",
        "train_months",
        "gru_hidden_dim",
        "gru_layers",
        "dropout",
        "lr",
        "epochs",
    }
    missing = sorted(required - set(raw.keys()))
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Missing config keys: {missing_keys}")

    raw_test_size = raw.get("test_size")
    config = EvalTemporalConfig(
        input_path=str(raw["input_path"]),
        output_dir=str(raw["output_dir"]),
        random_seed=int(raw["random_seed"]),
        model_type=str(raw.get("model_type", "gru")),
        encoder_model=str(raw["encoder_model"]),
        max_length=int(raw["max_length"]),
        batch_size=int(raw["batch_size"]),
        cache_embeddings=bool(raw["cache_embeddings"]),
        cache_dir=str(raw["cache_dir"]),
        train_months=int(raw["train_months"]),
        gru_hidden_dim=int(raw["gru_hidden_dim"]),
        gru_layers=int(raw["gru_layers"]),
        dropout=float(raw["dropout"]),
        lr=float(raw["lr"]),
        epochs=int(raw["epochs"]),
        threshold_mode=str(raw.get("threshold_mode", "fixed")),
        fixed_threshold=float(raw.get("fixed_threshold", 0.5)),
        calibration_metric=str(raw.get("calibration_metric", "balanced_accuracy")),
        test_size=None if raw_test_size is None else float(raw_test_size),
    )

    if not config.encoder_model.strip():
        raise ValueError("encoder_model must be non-empty")
    if config.max_length <= 0:
        raise ValueError("max_length must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.train_months < 1:
        raise ValueError("train_months must be >= 1")
    if config.model_type not in {"gru", "baseline_lr", "attention", "transformer"}:
        raise ValueError("model_type must be one of: gru, baseline_lr, attention, transformer")
    if config.gru_hidden_dim <= 0:
        raise ValueError("gru_hidden_dim must be > 0")
    if config.gru_layers <= 0:
        raise ValueError("gru_layers must be > 0")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if config.lr <= 0:
        raise ValueError("lr must be > 0")
    if config.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if config.threshold_mode not in {"fixed", "calibrate_first_eval", "calibrate_each_month"}:
        raise ValueError(
            "threshold_mode must be one of: fixed, calibrate_first_eval, calibrate_each_month"
        )
    if not 0.0 < config.fixed_threshold < 1.0:
        raise ValueError("fixed_threshold must be between 0 and 1")
    if config.calibration_metric not in {"youden_j", "balanced_accuracy"}:
        raise ValueError("calibration_metric must be one of: youden_j, balanced_accuracy")
    if config.test_size is not None and not 0.0 < config.test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1 when provided")

    return config


def load_train_e2e_config(path: str | Path) -> TrainE2EConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    required = {
        "input_path",
        "output_dir",
        "random_seed",
        "encoder_model",
        "max_length",
        "batch_size",
        "train_months",
        "gru_hidden_dim",
        "gru_layers",
        "dropout",
        "lr",
        "epochs",
        "test_size",
    }
    missing = sorted(required - set(raw.keys()))
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Missing config keys: {missing_keys}")

    config = TrainE2EConfig(
        input_path=str(raw["input_path"]),
        output_dir=str(raw["output_dir"]),
        random_seed=int(raw["random_seed"]),
        encoder_model=str(raw["encoder_model"]),
        max_length=int(raw["max_length"]),
        batch_size=int(raw["batch_size"]),
        train_months=int(raw["train_months"]),
        gru_hidden_dim=int(raw["gru_hidden_dim"]),
        gru_layers=int(raw["gru_layers"]),
        dropout=float(raw["dropout"]),
        lr=float(raw["lr"]),
        epochs=int(raw["epochs"]),
        test_size=float(raw["test_size"]),
        pooling=str(raw.get("pooling", "cls")),
        freeze_encoder=bool(raw.get("freeze_encoder", False)),
        pretrained_encoder_path=str(raw.get("pretrained_encoder_path", "")),
    )

    if not config.encoder_model.strip():
        raise ValueError("encoder_model must be non-empty")
    if config.max_length <= 0:
        raise ValueError("max_length must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.train_months < 1:
        raise ValueError("train_months must be >= 1")
    if config.gru_hidden_dim <= 0:
        raise ValueError("gru_hidden_dim must be > 0")
    if config.gru_layers <= 0:
        raise ValueError("gru_layers must be > 0")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if config.lr <= 0:
        raise ValueError("lr must be > 0")
    if config.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if not 0.0 < config.test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if config.pooling not in {"cls", "mean"}:
        raise ValueError("pooling must be one of: cls, mean")

    return config


def load_eval_e2e_config(path: str | Path) -> EvalE2EConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    required = {
        "input_path",
        "output_dir",
        "random_seed",
        "encoder_model",
        "max_length",
        "batch_size",
        "train_months",
        "checkpoint_path",
        "test_size",
    }
    missing = sorted(required - set(raw.keys()))
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Missing config keys: {missing_keys}")

    config = EvalE2EConfig(
        input_path=str(raw["input_path"]),
        output_dir=str(raw["output_dir"]),
        random_seed=int(raw["random_seed"]),
        encoder_model=str(raw["encoder_model"]),
        max_length=int(raw["max_length"]),
        batch_size=int(raw["batch_size"]),
        train_months=int(raw["train_months"]),
        checkpoint_path=str(raw["checkpoint_path"]),
        test_size=float(raw["test_size"]),
        pooling=str(raw.get("pooling", "cls")),
        threshold=float(raw.get("threshold", 0.5)),
    )

    if not config.encoder_model.strip():
        raise ValueError("encoder_model must be non-empty")
    if config.max_length <= 0:
        raise ValueError("max_length must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.train_months < 1:
        raise ValueError("train_months must be >= 1")
    if not config.checkpoint_path.strip():
        raise ValueError("checkpoint_path must be non-empty")
    if not 0.0 < config.test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if config.pooling not in {"cls", "mean"}:
        raise ValueError("pooling must be one of: cls, mean")
    if not 0.0 < config.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    return config
