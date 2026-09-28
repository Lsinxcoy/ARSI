"""RRSI L0-style annealed edit budget (arXiv:2609.24972 Eq.4).

b_t = ceil(b_min + (b_max-b_min) * 0.5 * (1 + cos(π t / T)))
Early rounds may bundle; late rounds must be attributable.
"""
from __future__ import annotations

import math


def edit_budget(
    t: int,
    T: int,
    bmin: int = 1,
    bmax: int = 4,
) -> int:
    """Atomic edits allowed in one candidate at round t of T."""
    T = max(1, int(T))
    t = max(0, min(int(t), T))
    bmin = max(1, int(bmin))
    bmax = max(bmin, int(bmax))
    frac = 0.5 * (1.0 + math.cos(math.pi * t / T))
    return int(math.ceil(bmin + (bmax - bmin) * frac))


def stall_flag(score_now: float, score_w_ago: float, delta: float) -> bool:
    """σ_t = 1[ Ŝ_t - Ŝ_{t-w} ≤ δ ] — progress within noise band."""
    return (float(score_now) - float(score_w_ago)) <= float(delta)


def unexercised_components(all_components: set[str], exercised: set[str]) -> list[str]:
    """U_t = K \\ T_t"""
    return sorted(set(all_components) - set(exercised))
