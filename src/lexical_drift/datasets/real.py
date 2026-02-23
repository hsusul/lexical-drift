from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_SAMPLE_LOCAL_COLUMNS = {"author_id", "month", "text", "label"}
REQUIRED_PREPARED_COLUMNS = {"author_id", "month_index", "text", "drift_label"}


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


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported input format: {path.suffix}. Use .csv or .jsonl")


def _normalize_real_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if REQUIRED_PREPARED_COLUMNS.issubset(frame.columns):
        normalized = pd.DataFrame(
            {
                "author_id": frame["author_id"].astype(str),
                "month_index": _to_month_index(frame["month_index"]),
                "text": frame["text"].astype(str),
                "drift_label": frame["drift_label"].astype(int),
            }
        )
        return normalized.sort_values(["author_id", "month_index"]).reset_index(drop=True)

    if REQUIRED_SAMPLE_LOCAL_COLUMNS.issubset(frame.columns):
        normalized = pd.DataFrame(
            {
                "author_id": frame["author_id"].astype(str),
                "month_index": _to_month_index(frame["month"]),
                "text": frame["text"].astype(str),
                "drift_label": frame["label"].astype(int),
            }
        )
        return normalized.sort_values(["author_id", "month_index"]).reset_index(drop=True)

    missing_sample = sorted(REQUIRED_SAMPLE_LOCAL_COLUMNS - set(frame.columns))
    missing_prepared = sorted(REQUIRED_PREPARED_COLUMNS - set(frame.columns))
    raise ValueError(
        "Dataset schema mismatch. "
        f"Expected sample columns (missing: {', '.join(missing_sample)}) "
        "or prepared columns "
        f"(missing: {', '.join(missing_prepared)})."
    )


def prepare_real_dataset(
    *,
    input_path: str | Path,
    out_path: str | Path,
) -> Path:
    source_path = Path(input_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Real dataset path not found: {source_path}")

    frame = _read_table(source_path)
    normalized = _normalize_real_frame(frame)
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        try:
            normalized.to_parquet(output_path, index=False)
        except ImportError as exc:
            raise ImportError(
                "Writing parquet requires pyarrow or fastparquet. "
                "Install one of them or choose a .csv output path."
            ) from exc
    else:
        normalized.to_csv(output_path, index=False)
    return output_path


def load_real_dataset(
    *,
    name: str,
    path: str | Path,
) -> pd.DataFrame:
    dataset_name = name.strip().lower()
    if dataset_name not in {"sample_local", "prepared_local"}:
        raise ValueError(f"Unsupported real dataset name: {name}")

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Real dataset path not found: {source_path}")
    if dataset_name == "prepared_local":
        if source_path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(source_path)
        else:
            frame = _read_table(source_path)
        return _normalize_real_frame(frame)

    frame = _read_table(source_path)
    return _normalize_real_frame(frame)
