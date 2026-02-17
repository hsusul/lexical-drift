from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from lexical_drift.config import TrainConfig
from lexical_drift.models.baseline import build_baseline_model, evaluate_model
from lexical_drift.utils import ensure_dir

REQUIRED_COLUMNS = {"author_id", "month_index", "text", "drift_label"}


def run_training(config: TrainConfig) -> dict[str, float]:
    data_path = Path(config.input_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {data_path}")

    frame = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    X = frame["text"].astype(str).tolist()
    y = frame["drift_label"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y,
    )

    model = build_baseline_model(
        max_features=config.max_features,
        c_value=config.C,
        random_seed=config.random_seed,
    )
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    output_dir = ensure_dir(config.output_dir)
    model_path = output_dir / "baseline.joblib"
    metadata_path = output_dir / "metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_type": "tfidf_logreg",
        "input_path": str(data_path),
        "model_path": str(model_path),
        "metrics": metrics,
        "config": asdict(config),
        "n_rows": int(len(frame)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
    }
