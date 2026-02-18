from __future__ import annotations

import pandas as pd

from lexical_drift.training.train_temporal import (
    SMALL_FILE_THRESHOLD_BYTES,
    compute_dataset_fingerprint,
)


def test_dataset_fingerprint_changes_with_text_content(tmp_path) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"

    base_rows = [
        {"author_id": "a1", "month_index": 0, "text": "first text", "drift_label": 0},
        {"author_id": "a1", "month_index": 1, "text": "second text", "drift_label": 0},
        {"author_id": "a2", "month_index": 0, "text": "other text", "drift_label": 1},
        {"author_id": "a2", "month_index": 1, "text": "more text", "drift_label": 1},
    ]
    changed_rows = [
        {"author_id": "a1", "month_index": 0, "text": "first text", "drift_label": 0},
        {"author_id": "a1", "month_index": 1, "text": "changed text", "drift_label": 0},
        {"author_id": "a2", "month_index": 0, "text": "other text", "drift_label": 1},
        {"author_id": "a2", "month_index": 1, "text": "more text", "drift_label": 1},
    ]

    pd.DataFrame(base_rows).to_csv(csv_a, index=False)
    pd.DataFrame(changed_rows).to_csv(csv_b, index=False)

    fingerprint_a = compute_dataset_fingerprint(csv_a)
    fingerprint_a_repeat = compute_dataset_fingerprint(csv_a)
    fingerprint_b = compute_dataset_fingerprint(csv_b)

    assert fingerprint_a == fingerprint_a_repeat
    assert fingerprint_a != fingerprint_b


def test_dataset_fingerprint_large_file_sampled_branch(tmp_path) -> None:
    path = tmp_path / "large.csv"
    payload = (b"author_id,month_index,text,drift_label\n" + b"x,0,alpha,0\n") * (
        SMALL_FILE_THRESHOLD_BYTES // 16 + 1000
    )
    path.write_bytes(payload)

    first = compute_dataset_fingerprint(path)
    second = compute_dataset_fingerprint(path)
    assert first == second

    modified = bytearray(payload)
    modified[0] = ord("z")
    path.write_bytes(modified)
    changed = compute_dataset_fingerprint(path)
    assert changed != first
