"""Feedback-space co-evolution rules (arXiv:2608.10299 second-pass W1–W5).

W1 ECHO actionability: only update diagnostician if following advice helped.
W2 ARCO: step scores must sum to outcome.
W3 PEBBLE: relabel history when the metric/reward model changes.
W4 Hint fade: scaffolding exits as capability rises.
W5 R*: multi-critic agreement before changing the metric.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence


@dataclass
class ActionabilityGate:
    ok: bool
    delta_after_advice: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def actionability_gate(
    outcome_before: float,
    outcome_after_advice: float,
    eps: float = 0.0,
) -> ActionabilityGate:
    """W1 ECHO: critic/metric update only if acting on advice improved outcome."""
    d = float(outcome_after_advice) - float(outcome_before)
    ok = d > eps
    return ActionabilityGate(
        ok=ok,
        delta_after_advice=round(d, 6),
        reason="advice_helped_update_critic" if ok else "advice_did_not_help_keep_critic",
    )


def step_outcome_consistency(
    step_scores: Sequence[float],
    outcome: float,
    tol: float = 1e-3,
) -> tuple[bool, str]:
    """W2 ARCO: sum(step) ≈ outcome, else metric untrustworthy."""
    if not step_scores:
        return True, "no_steps"
    s = sum(float(x) for x in step_scores)
    ok = abs(s - float(outcome)) <= tol
    return ok, "consistent" if ok else f"step_sum={s:.4f}!={outcome:.4f}"


def relabel_on_metric_change(
    old_preds: Sequence[str],
    metric_version_before: str,
    metric_version_after: str,
) -> Optional[list[str]]:
    """W3 PEBBLE: if metric changed, old verdicts must be re-scored (placeholder marks)."""
    if metric_version_before == metric_version_after:
        return None
    return [f"relabel:{p}" for p in old_preds]


def hint_fade(
    capability: float,
    hint_strength: float = 1.0,
    fade_start: float = 0.6,
    fade_end: float = 0.9,
) -> float:
    """W4: scaffold strength decays as capability rises (anti-dependency)."""
    c = float(capability)
    if c <= fade_start:
        return float(hint_strength)
    if c >= fade_end:
        return 0.0
    k = (fade_end - c) / (fade_end - fade_start)
    return round(float(hint_strength) * k, 4)


def metric_update_allowed(
    critic_votes: Sequence[bool],
    agreement_frac: float = 0.67,
) -> tuple[bool, str]:
    """W5 R*: only revise metric when critics agree."""
    n = len(critic_votes or [])
    if n == 0:
        return False, "no_critic_votes"
    frac = sum(1 for v in critic_votes if v) / n
    ok = frac >= agreement_frac
    return ok, "metric_update_ok" if ok else f"disagreement_{frac:.2f}"
