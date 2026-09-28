"""P-h Ignition test (AIDE² second-pass).

Can a discovered/improved component act as the **outer loop** (rewrite the
improver itself)? Ignition = self-improvement of the improvement mechanism.
ARSI: take evolved Evolver/bandit policy and drive one outer round with it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


@dataclass
class IgnitionResult:
    ignited: bool
    outer_gain: float = 0.0
    baseline_gain: float = 0.0
    sample_efficiency_hint: bool = False
    reason: str = ""
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def ignition_test(
    *,
    incumbent_grade: float,
    grade_after_discovered_outer: float,
    grade_after_baseline_outer: float,
    steps_discovered: int,
    steps_baseline: int,
    eps: float = 1e-6,
) -> IgnitionResult:
    """AIDE² §3.6: discovered agent as outer loop vs human/baseline outer.

    ignited if discovered outer does not degrade and shows possible sample-efficiency
    (same or fewer steps for same/better gain). Definitive claims need more seeds.
    """
    g0 = float(incumbent_grade)
    gd = float(grade_after_discovered_outer)
    gb = float(grade_after_baseline_outer)
    gain_d = gd - g0
    gain_b = gb - g0
    no_degrade = gd >= g0 - eps
    # sample efficiency hint: steps to reach gain, lower is better
    se = False
    if steps_discovered > 0 and steps_baseline > 0 and gain_d >= gain_b - eps:
        se = steps_discovered <= steps_baseline
    ignited = no_degrade and gain_d >= -eps
    return IgnitionResult(
        ignited=bool(ignited),
        outer_gain=round(gain_d, 6),
        baseline_gain=round(gain_b, 6),
        sample_efficiency_hint=bool(se),
        reason="ignited" if ignited else "not_ignited",
        detail={
            "incumbent": g0,
            "discovered": gd,
            "baseline": gb,
            "steps_discovered": steps_discovered,
            "steps_baseline": steps_baseline,
            "note": "definitive_needs_more_seeds_aide2",
        },
    )


def run_outer_round(
    outer_fn: Callable[[Any], Any],
    incumbent: Any,
    grade_fn: Callable[[Any], float],
) -> dict:
    """One outer-loop rewrite attempt using a (possibly discovered) modifier."""
    proposal = outer_fn(incumbent)
    g_before = grade_fn(incumbent)
    g_after = grade_fn(proposal)
    accepted = g_after >= g_before
    return {
        "accepted": bool(accepted),
        "g_before": g_before,
        "g_after": g_after,
        "proposal": proposal,
        "note": "outer_loop_rewrite",
    }
