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
TOPIC_GROUPS = [
    ["analysis", "research", "evidence", "study", "experiment", "result"],
    ["language", "grammar", "style", "dialogue", "narrative", "clarity"],
    ["signal", "feature", "pattern", "baseline", "dataset", "token"],
    ["question", "response", "document", "context", "topic", "report"],
]
DIFFICULTY_PRESETS = {
    "easy": {
        "drift_strength": 1.15,
        "noise_strength": 0.22,
        "global_event_strength": 0.15,
        "topic_shift_strength": 0.75,
    },
    "hard": {
        "drift_strength": 0.55,
        "noise_strength": 0.40,
        "global_event_strength": 0.33,
        "topic_shift_strength": 0.32,
    },
}


def _normalize_probs(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, None)
    return clipped / clipped.sum()


def _resolve_knobs(
    difficulty: str,
    drift_strength: float | None,
    noise_strength: float | None,
    global_event_strength: float | None,
    topic_shift_strength: float | None,
) -> dict[str, float]:
    if difficulty not in DIFFICULTY_PRESETS:
        options = ", ".join(sorted(DIFFICULTY_PRESETS))
        raise ValueError(f"difficulty must be one of: {options}")

    preset = DIFFICULTY_PRESETS[difficulty]
    return {
        "drift_strength": float(
            preset["drift_strength"] if drift_strength is None else drift_strength
        ),
        "noise_strength": float(
            preset["noise_strength"] if noise_strength is None else noise_strength
        ),
        "global_event_strength": float(
            preset["global_event_strength"]
            if global_event_strength is None
            else global_event_strength
        ),
        "topic_shift_strength": float(
            preset["topic_shift_strength"] if topic_shift_strength is None else topic_shift_strength
        ),
    }


def _build_topic_word_probs(rng: np.random.Generator) -> np.ndarray:
    topic_word_probs = []
    for group in TOPIC_GROUPS:
        weights = np.full(len(BASE_VOCAB), 0.2, dtype=np.float64)
        for token in group:
            token_idx = BASE_VOCAB.index(token)
            weights[token_idx] += float(rng.uniform(1.2, 2.0))
        topic_word_probs.append(_normalize_probs(weights))
    return np.asarray(topic_word_probs, dtype=np.float64)


def _make_sentence(
    rng: np.random.Generator,
    *,
    word_probs: np.ndarray,
    punctuation_probs: np.ndarray,
    min_len: int,
    max_len: int,
    filler_rate: float,
    clause_rate: float,
) -> str:
    length = int(rng.integers(min_len, max_len + 1))
    tokens: list[str] = []
    for _ in range(length):
        if rng.random() < filler_rate:
            tokens.append(str(rng.choice(FILLERS)))
        else:
            tokens.append(str(rng.choice(BASE_VOCAB, p=word_probs)))

    sentence = " ".join(tokens).capitalize()

    if rng.random() < clause_rate:
        clause_len = int(rng.integers(3, 7))
        clause_tokens = [str(rng.choice(BASE_VOCAB, p=word_probs)) for _ in range(clause_len)]
        connector = str(rng.choice(CONNECTORS))
        sentence = f"{sentence}, {connector} {' '.join(clause_tokens)}"

    punctuation = str(rng.choice([".", "!", "?"], p=punctuation_probs))
    return sentence + punctuation


def _author_text(
    rng: np.random.Generator,
    *,
    month_index: int,
    months: int,
    drift_label: int,
    topic_word_probs: np.ndarray,
    author_topic_mix: np.ndarray,
    author_topic_direction: np.ndarray,
    month_topic_events: np.ndarray,
    month_style_events: np.ndarray,
    base_sentence_len: float,
    base_num_sentences: float,
    base_filler_rate: float,
    base_clause_rate: float,
    base_exclaim_rate: float,
    drift_slope: float,
    topic_shift_slope: float,
    drift_strength: float,
    topic_shift_strength: float,
    noise_strength: float,
) -> str:
    progress = month_index / max(months - 1, 1)
    month_event = float(month_style_events[month_index])

    drift_latent = (
        drift_strength * drift_slope * progress
        + 0.35 * month_event
        + float(rng.normal(0.0, noise_strength * 0.22))
    )
    if drift_label == 0:
        drift_latent *= 0.72

    topic_offset = (
        month_topic_events[month_index]
        + float(rng.normal(0.0, noise_strength * 0.05)) * author_topic_direction
    )
    topic_mix = (
        author_topic_mix + topic_shift_strength * topic_shift_slope * progress * topic_offset
    )
    topic_mix = _normalize_probs(topic_mix)
    word_probs = _normalize_probs(topic_mix @ topic_word_probs)

    sentence_center = base_sentence_len - 1.9 * drift_latent + float(rng.normal(0.0, 0.6))
    min_len = int(np.clip(round(sentence_center - 2.5), 4, 24))
    max_len = int(np.clip(round(sentence_center + 3.0), min_len + 2, 30))

    sentence_count = int(
        np.clip(
            round(base_num_sentences + 0.45 * month_event + rng.normal(0.0, 0.5)),
            1,
            5,
        )
    )
    filler_rate = float(
        np.clip(
            base_filler_rate + 0.08 * drift_latent + 0.04 * month_event + rng.normal(0.0, 0.02),
            0.01,
            0.45,
        )
    )
    clause_rate = float(
        np.clip(
            base_clause_rate - 0.15 * drift_latent + rng.normal(0.0, noise_strength * 0.03),
            0.08,
            0.85,
        )
    )
    exclaim_rate = float(
        np.clip(
            base_exclaim_rate + 0.03 * month_event + rng.normal(0.0, 0.015),
            0.02,
            0.30,
        )
    )
    question_rate = float(
        np.clip(
            0.10 + 0.015 * month_event + rng.normal(0.0, 0.01),
            0.03,
            0.22,
        )
    )
    period_rate = max(0.05, 1.0 - exclaim_rate - question_rate)
    punctuation_probs = _normalize_probs(
        np.asarray([period_rate, exclaim_rate, question_rate], dtype=np.float64)
    )

    parts = [
        _make_sentence(
            rng,
            word_probs=word_probs,
            punctuation_probs=punctuation_probs,
            min_len=min_len,
            max_len=max_len,
            filler_rate=filler_rate,
            clause_rate=clause_rate,
        )
        for _ in range(sentence_count)
    ]

    return " ".join(parts)


def generate_synthetic_dataset(
    n_authors: int,
    months: int,
    random_seed: int = 42,
    difficulty: str = "easy",
    drift_strength: float | None = None,
    noise_strength: float | None = None,
    global_event_strength: float | None = None,
    topic_shift_strength: float | None = None,
) -> pd.DataFrame:
    if n_authors <= 0:
        raise ValueError("n_authors must be > 0")
    if months <= 1:
        raise ValueError("months must be > 1")

    knobs = _resolve_knobs(
        difficulty=difficulty,
        drift_strength=drift_strength,
        noise_strength=noise_strength,
        global_event_strength=global_event_strength,
        topic_shift_strength=topic_shift_strength,
    )
    rng = np.random.default_rng(random_seed)
    topic_word_probs = _build_topic_word_probs(rng)
    n_topics = int(topic_word_probs.shape[0])

    month_style_events = rng.normal(0.0, knobs["global_event_strength"], size=months)
    month_topic_events = rng.normal(0.0, knobs["global_event_strength"], size=(months, n_topics))
    drift_labels = rng.integers(0, 2, size=n_authors)
    if n_authors > 1 and int(drift_labels.min()) == int(drift_labels.max()):
        drift_labels[0] = 0
        drift_labels[-1] = 1

    rows: list[dict[str, object]] = []

    for author_idx, drift_label in enumerate(drift_labels.tolist()):
        author_id = f"author_{author_idx:03d}"

        author_topic_mix = rng.dirichlet(np.full(n_topics, 1.3))
        author_topic_direction = rng.normal(0.0, 1.0, size=n_topics)
        author_topic_direction = author_topic_direction - author_topic_direction.mean()
        author_topic_direction = _normalize_probs(np.abs(author_topic_direction))

        base_sentence_len = float(rng.normal(10.8, 1.8))
        base_num_sentences = float(rng.uniform(2.0, 3.8))
        base_filler_rate = float(rng.uniform(0.03, 0.12))
        base_clause_rate = float(rng.uniform(0.30, 0.70))
        base_exclaim_rate = float(rng.uniform(0.04, 0.12))

        if drift_label == 1:
            drift_slope = float(rng.normal(0.95, knobs["noise_strength"] * 0.45))
            topic_shift_slope = float(rng.normal(0.75, knobs["noise_strength"] * 0.35))
        else:
            drift_slope = float(rng.normal(0.45, knobs["noise_strength"] * 0.50))
            topic_shift_slope = float(rng.normal(0.38, knobs["noise_strength"] * 0.40))

        for month_index in range(months):
            text = _author_text(
                rng,
                month_index=month_index,
                months=months,
                drift_label=drift_label,
                topic_word_probs=topic_word_probs,
                author_topic_mix=author_topic_mix,
                author_topic_direction=author_topic_direction,
                month_topic_events=month_topic_events,
                month_style_events=month_style_events,
                base_sentence_len=base_sentence_len,
                base_num_sentences=base_num_sentences,
                base_filler_rate=base_filler_rate,
                base_clause_rate=base_clause_rate,
                base_exclaim_rate=base_exclaim_rate,
                drift_slope=drift_slope,
                topic_shift_slope=topic_shift_slope,
                drift_strength=knobs["drift_strength"],
                topic_shift_strength=knobs["topic_shift_strength"],
                noise_strength=knobs["noise_strength"],
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
    difficulty: str = "easy",
    drift_strength: float | None = None,
    noise_strength: float | None = None,
    global_event_strength: float | None = None,
    topic_shift_strength: float | None = None,
) -> Path:
    frame = generate_synthetic_dataset(
        n_authors=n_authors,
        months=months,
        random_seed=random_seed,
        difficulty=difficulty,
        drift_strength=drift_strength,
        noise_strength=noise_strength,
        global_event_strength=global_event_strength,
        topic_shift_strength=topic_shift_strength,
    )
    output = ensure_parent_dir(out_path)
    frame.to_csv(output, index=False)
    return output
