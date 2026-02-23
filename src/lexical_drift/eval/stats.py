from __future__ import annotations

import math

import numpy as np


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_t_test(deltas: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(deltas, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 2:
        return {"t_stat": None, "p_value": None, "n": n}

    mean_delta = float(values.mean())
    std_delta = float(values.std(ddof=1))
    if std_delta <= 0.0:
        t_stat = 0.0
        p_value = 1.0
        return {"t_stat": t_stat, "p_value": p_value, "n": n}

    t_stat = mean_delta / (std_delta / math.sqrt(float(n)))

    p_value: float
    try:
        from scipy import stats as scipy_stats  # type: ignore

        p_value = float(2.0 * scipy_stats.t.sf(abs(t_stat), df=n - 1))
    except Exception:
        # Fallback approximation using standard normal when SciPy is unavailable.
        p_value = float(2.0 * (1.0 - _normal_cdf(abs(t_stat))))

    return {"t_stat": float(t_stat), "p_value": p_value, "n": n}


def bootstrap_ci(
    deltas: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float | int | None]:
    values = np.asarray(deltas, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {"mean": None, "low": None, "high": None, "n": n}

    mean = float(values.mean())
    if n == 1:
        return {"mean": mean, "low": mean, "high": mean, "n": n}

    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        sampled = rng.choice(values, size=n, replace=True)
        draws[i] = sampled.mean()

    lower_q = float(alpha / 2.0)
    upper_q = float(1.0 - alpha / 2.0)
    low = float(np.quantile(draws, lower_q))
    high = float(np.quantile(draws, upper_q))
    return {"mean": mean, "low": low, "high": high, "n": n}
