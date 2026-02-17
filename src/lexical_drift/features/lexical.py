from __future__ import annotations

import re
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[A-Za-z_']+")


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def type_token_ratio(tokens: Iterable[str]) -> float:
    token_list = list(tokens)
    if not token_list:
        return 0.0
    return len(set(token_list)) / len(token_list)


def lexical_summary(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    return {
        "num_tokens": float(len(tokens)),
        "type_token_ratio": type_token_ratio(tokens),
    }
