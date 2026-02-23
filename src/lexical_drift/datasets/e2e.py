from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MonthlySequenceBatch:
    author_ids: list[str]
    texts: list[list[str]]
    labels: np.ndarray


def build_sequence_batch(
    *,
    author_ids: list[str],
    sequences_texts: list[list[str]],
    labels: np.ndarray,
    indices: list[int],
    max_months: int,
) -> MonthlySequenceBatch:
    batch_author_ids = [author_ids[index] for index in indices]
    batch_texts = [sequences_texts[index][:max_months] for index in indices]
    batch_labels = labels[np.asarray(indices, dtype=np.int64)].astype(np.float32)
    return MonthlySequenceBatch(
        author_ids=batch_author_ids,
        texts=batch_texts,
        labels=batch_labels,
    )
