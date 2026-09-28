"""RRSI noise-adjusted floor δ (arXiv:2609.24972 §3.3 / App.D).

Calibrate by repeatedly evaluating the unchanged base harness H0.
Never use held-out/OOD to set δ.
"""
from __future__ import annotations

import math
from typing import Sequence


def calibrate_delta(
    baseline_scores: Sequence[float],
    method: str = "range",
    min_n: int = 3,
) -> float:
    """Empirical noise band from repeated H0 scores.

    method='range' → (max-min)/2   (conservative, matches '3 of 178 passes' style)
    method='std'   → 2 * sample std
    """
    xs = [float(x) for x in baseline_scores or []]
    if len(xs) < max(2, min_n):
        return 0.02  # conservative default until calibrated
    if method == "std":
        n = len(xs)
        mu = sum(xs) / n
        var = sum((x - mu) ** 2 for x in xs) / max(1, n - 1)
        return max(1e-6, 2.0 * math.sqrt(var))
    return max(1e-6, (max(xs) - min(xs)) / 2.0)


def passes_floor(score_hat: float, s_star: float, delta: float) -> bool:
    """Ŝ(H') ≥ S* − δ"""
    return float(score_hat) >= float(s_star) - float(delta)
