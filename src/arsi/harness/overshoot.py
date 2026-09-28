"""P-i Overshoot / past-best detection (GAI second-pass empirical).

Paper: 58% improve on first valid attempt, but 78% of searches continue past
their best — candidate-self search runs away. Flag when we keep editing after
the incumbent already peaked.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence


@dataclass
class OvershootReport:
    overshoot: bool
    steps_past_best: int = 0
    best_index: int = -1
    best_score: float = 0.0
    last_score: float = 0.0
    advice: str = ""
    note: str = "gai_past_best_search"

    def to_dict(self) -> dict:
        return asdict(self)


def detect_overshoot(
    scores: Sequence[float],
    patience: int = 2,
    eps: float = 1e-9,
) -> OvershootReport:
    """scores: incumbent-grade trajectory. Overshoot if we keep going ≥patience past best."""
    xs = [float(s) for s in (scores or [])]
    if len(xs) < 2:
        return OvershootReport(False, advice="insufficient_history")
    best_i = max(range(len(xs)), key=lambda i: xs[i])
    best = xs[best_i]
    last = xs[-1]
    past = len(xs) - 1 - best_i
    overshoot = past >= patience and last < best - eps
    advice = "continue" if not overshoot else "rollback_to_best_or_stop"
    return OvershootReport(
        overshoot=bool(overshoot),
        steps_past_best=past,
        best_index=best_i,
        best_score=best,
        last_score=last,
        advice=advice,
    )


def should_stop_editing(scores: Sequence[float], patience: int = 2) -> tuple[bool, str]:
    r = detect_overshoot(scores, patience=patience)
    return r.overshoot, r.advice
