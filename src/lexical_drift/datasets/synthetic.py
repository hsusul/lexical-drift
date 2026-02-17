from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lexical_drift.utils import ensure_parent_dir

BASE_VOCAB = [
    "analysis",
    "study",
    "language",
    "signal",
    "context",
    "change",
    "topic",
    "dialogue",
    "token",
    "grammar",
    "style",
    "sample",
    "review",
    "result",
    "pattern",
    "author",
    "month",
    "narrative",
    "clarity",
    "insight",
    "evidence",
    "question",
    "response",
    "document",
    "research",
    "baseline",
    "feature",
    "dataset",
    "report",
    "experiment",
]

FILLERS = ["um", "uh", "like", "you_know", "basically"]
CONNECTORS = ["because", "while", "although", "when"]


def _make_sentence(
    rng: np.random.Generator,
    vocab: list[str],
    *,
    min_len: int,
    max_len: int,
    filler_rate: float,
    with_clause: bool,
) -> str:
    length = int(rng.integers(min_len, max_len + 1))
    tokens: list[str] = []
    for _ in range(length):
        if rng.random() < filler_rate:
            tokens.append(str(rng.choice(FILLERS)))
        else:
            tokens.append(str(rng.choice(vocab)))

    sentence = " ".join(tokens).capitalize()

    if with_clause and rng.random() < 0.6:
        clause_len = int(rng.integers(3, 7))
        clause_tokens = [str(rng.choice(vocab)) for _ in range(clause_len)]
        connector = str(rng.choice(CONNECTORS))
        sentence = f"{sentence}, {connector} {' '.join(clause_tokens)}"

    return sentence + str(rng.choice([".", ".", "!"]))


def _author_text(
    rng: np.random.Generator,
    month_index: int,
    months: int,
    drift_label: int,
    normal_vocab: list[str],
    drift_vocab: list[str],
) -> str:
    later_phase = month_index >= months // 2
    apply_drift = drift_label == 1 and later_phase

    if apply_drift:
        # Drift injection: fewer unique words, shorter text, more fillers.
        n_sentences = int(rng.integers(1, 3))
        parts = [
            _make_sentence(
                rng,
                drift_vocab,
                min_len=4,
                max_len=7,
                filler_rate=0.35,
                with_clause=False,
            )
            for _ in range(n_sentences)
        ]
    else:
        n_sentences = int(rng.integers(2, 5))
        parts = [
            _make_sentence(
                rng,
                normal_vocab,
                min_len=7,
                max_len=14,
                filler_rate=0.05,
                with_clause=True,
            )
            for _ in range(n_sentences)
        ]

    return " ".join(parts)


def generate_synthetic_dataset(
    n_authors: int,
    months: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    if n_authors <= 0:
        raise ValueError("n_authors must be > 0")
    if months <= 1:
        raise ValueError("months must be > 1")

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []

    for author_idx in range(n_authors):
        author_id = f"author_{author_idx:03d}"
        drift_label = int(rng.random() < 0.5)

        normal_vocab = rng.choice(BASE_VOCAB, size=18, replace=False).tolist()
        drift_vocab = rng.choice(normal_vocab, size=5, replace=False).tolist()

        for month_index in range(months):
            text = _author_text(
                rng,
                month_index=month_index,
                months=months,
                drift_label=drift_label,
                normal_vocab=normal_vocab,
                drift_vocab=drift_vocab,
            )
            rows.append(
                {
                    "author_id": author_id,
                    "month_index": month_index,
                    "text": text,
                    "drift_label": drift_label,
                }
            )

    return pd.DataFrame(rows, columns=["author_id", "month_index", "text", "drift_label"])


def save_synthetic_dataset(
    out_path: str | Path,
    *,
    n_authors: int,
    months: int,
    random_seed: int = 42,
) -> Path:
    frame = generate_synthetic_dataset(n_authors=n_authors, months=months, random_seed=random_seed)
    output = ensure_parent_dir(out_path)
    frame.to_csv(output, index=False)
    return output
