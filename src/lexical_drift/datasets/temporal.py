from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"author_id", "month_index", "text", "drift_label"}


def build_author_sequences_with_months(
    frame: pd.DataFrame,
) -> tuple[list[str], list[list[str]], list[list[int]], np.ndarray]:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    authors: list[str] = []
    sequences_texts: list[list[str]] = []
    sequences_months: list[list[int]] = []
    labels: list[int] = []

    for author_id in sorted(frame["author_id"].astype(str).unique().tolist()):
        author_df = frame[frame["author_id"].astype(str) == author_id].sort_values(
            "month_index", ascending=True
        )
        label_values = author_df["drift_label"].astype(int).unique().tolist()
        if len(label_values) != 1:
            raise ValueError(f"Author {author_id} has inconsistent drift labels")

        authors.append(author_id)
        sequences_texts.append(author_df["text"].astype(str).tolist())
        sequences_months.append(author_df["month_index"].astype(int).tolist())
        labels.append(int(label_values[0]))

    return authors, sequences_texts, sequences_months, np.asarray(labels, dtype=np.int64)


def build_author_sequences(frame: pd.DataFrame) -> tuple[list[str], list[list[str]], np.ndarray]:
    authors, sequences_texts, _sequences_months, labels = build_author_sequences_with_months(frame)
    return authors, sequences_texts, labels
