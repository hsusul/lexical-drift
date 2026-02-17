from __future__ import annotations

from pathlib import Path

import joblib


def predict_text(model_path: str | Path, text: str) -> dict[str, float | int]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)
    label = int(model.predict([text])[0])

    score = 0.0
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba([text])[0][1])

    return {
        "drift_label": label,
        "drift_score": score,
    }
