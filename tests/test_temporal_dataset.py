from __future__ import annotations

import pandas as pd

from lexical_drift.datasets.temporal import (
    build_author_sequences,
    build_author_sequences_with_months,
)


def test_build_author_sequences_orders_months() -> None:
    frame = pd.DataFrame(
        [
            {"author_id": "a2", "month_index": 1, "text": "later", "drift_label": 0},
            {"author_id": "a1", "month_index": 1, "text": "b", "drift_label": 1},
            {"author_id": "a1", "month_index": 0, "text": "a", "drift_label": 1},
            {"author_id": "a2", "month_index": 0, "text": "early", "drift_label": 0},
        ]
    )

    authors, sequences_texts, labels = build_author_sequences(frame)
    assert authors == ["a1", "a2"]
    assert sequences_texts == [["a", "b"], ["early", "later"]]
    assert labels.tolist() == [1, 0]

    authors2, _texts2, months2, labels2 = build_author_sequences_with_months(frame)
    assert authors2 == ["a1", "a2"]
    assert months2 == [[0, 1], [0, 1]]
    assert labels2.tolist() == [1, 0]
