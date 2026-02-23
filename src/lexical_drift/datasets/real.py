from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_SAMPLE_LOCAL_COLUMNS = {"author_id", "month", "text", "label"}


def _to_month_index(month_series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(month_series, errors="coerce")
    if not numeric.isna().any():
        return numeric.astype(int)

    month_as_datetime = pd.to_datetime(month_series, errors="coerce")
    if not month_as_datetime.isna().any():
        unique_months = sorted(month_as_datetime.unique())
        month_rank = {month: index for index, month in enumerate(unique_months)}
        return month_as_datetime.map(month_rank).astype(int)

    unique_values = sorted(month_series.astype(str).unique())
    month_rank = {value: index for index, value in enumerate(unique_values)}
    return month_series.astype(str).map(month_rank).astype(int)


def load_real_dataset(
    *,
    name: str,
    path: str | Path,
) -> pd.DataFrame:
    dataset_name = name.strip().lower()
    if dataset_name != "sample_local":
        raise ValueError(f"Unsupported real dataset name: {name}")

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Real dataset path not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = REQUIRED_SAMPLE_LOCAL_COLUMNS - set(frame.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    normalized = pd.DataFrame(
        {
            "author_id": frame["author_id"].astype(str),
            "month_index": _to_month_index(frame["month"]),
            "text": frame["text"].astype(str),
            "drift_label": frame["label"].astype(int),
        }
    )
    normalized = normalized.sort_values(["author_id", "month_index"]).reset_index(drop=True)
    return normalized
