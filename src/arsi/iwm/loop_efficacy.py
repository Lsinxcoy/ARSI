"""LoopEfficacy / LoopTrial — Q3: did self-modification help?

η (or any self-metric) may only move on evidence. No hardcoded "dream halves η".
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

VERDICT_HELPED = "helped"
VERDICT_NEUTRAL = "neutral"
VERDICT_HURT = "hurt"
VERDICT_UNKNOWN = "unknown"


@dataclass
class LoopTrial:
    trial_id: str = ""
    intervention: str = ""
    metric_name: str = "prediction_error"
    before: float = 0.0
    after: float = 0.0
    delta: float = 0.0
    verdict: str = VERDICT_UNKNOWN
    evidence_refs: list[str] = field(default_factory=list)
    eta_before: Optional[float] = None
    eta_after: Optional[float] = None
    notes: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.trial_id:
            self.trial_id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)


class LoopEfficacy:
    """Tracks whether interventions (dream, empower, policy change) help."""

    def __init__(self, improve_eps: float = 0.02, hurt_eps: float = 0.02):
        self.improve_eps = improve_eps
        self.hurt_eps = hurt_eps
        self._trials: list[LoopTrial] = []
        self._open: dict[str, LoopTrial] = {}

    def begin(
        self,
        intervention: str,
        before: float,
        metric_name: str = "prediction_error",
        eta_before: Optional[float] = None,
        evidence_refs: Optional[list[str]] = None,
        notes: Optional[dict] = None,
    ) -> LoopTrial:
        trial = LoopTrial(
            intervention=intervention,
            metric_name=metric_name,
            before=float(before),
            eta_before=eta_before,
            evidence_refs=list(evidence_refs or []),
            notes=dict(notes or {}),
        )
        self._open[trial.trial_id] = trial
        return trial

    def complete(
        self,
        trial_id: str,
        after: float,
        eta_after: Optional[float] = None,
        evidence_refs: Optional[list[str]] = None,
        notes: Optional[dict] = None,
    ) -> LoopTrial:
        trial = self._open.pop(trial_id, None)
        if trial is None:
            trial = LoopTrial(intervention="unknown", before=float(after))
        trial.after = float(after)
        trial.delta = round(trial.after - trial.before, 6)
        if eta_after is not None:
            trial.eta_after = eta_after
        if evidence_refs:
            trial.evidence_refs.extend(evidence_refs)
        if notes:
            trial.notes.update(notes)

        if trial.delta <= -self.improve_eps:
            trial.verdict = VERDICT_HELPED
        elif trial.delta >= self.hurt_eps:
            trial.verdict = VERDICT_HURT
        else:
            trial.verdict = VERDICT_NEUTRAL
        trial.notes["verdict_rule"] = (
            f"helped if delta<=-{self.improve_eps}; hurt if delta>={self.hurt_eps}"
        )
        self._trials.append(trial)
        return trial

    def record_verdict(
        self,
        intervention: str,
        before: float,
        after: float,
        evidence_refs: Optional[list[str]] = None,
        **kwargs,
    ) -> LoopTrial:
        trial = self.begin(
            intervention=intervention,
            before=before,
            eta_before=kwargs.get("eta_before"),
            evidence_refs=evidence_refs,
            notes={k: v for k, v in kwargs.items() if k != "eta_before"},
        )
        return self.complete(trial.trial_id, after=after, eta_after=kwargs.get("eta_after"))

    def recent_verdicts(self, intervention: Optional[str] = None, n: int = 10) -> list[str]:
        rows = [t for t in self._trials if intervention is None or t.intervention == intervention]
        return [t.verdict for t in rows[-n:]]

    def intervention_summary(self, intervention: str) -> dict:
        rows = [t for t in self._trials if t.intervention == intervention]
        if not rows:
            return {
                "intervention": intervention,
                "count": 0,
                "helped": 0,
                "neutral": 0,
                "hurt": 0,
                "last_verdict": VERDICT_UNKNOWN,
                "allowed_default": True,
            }
        helped = sum(1 for t in rows if t.verdict == VERDICT_HELPED)
        hurt = sum(1 for t in rows if t.verdict == VERDICT_HURT)
        neutral = sum(1 for t in rows if t.verdict == VERDICT_NEUTRAL)
        last = rows[-1].verdict
        # Continuous neutral/hurt → forbid default use of this intervention
        recent = [t.verdict for t in rows[-3:]]
        allowed = not (len(recent) >= 2 and all(v in (VERDICT_NEUTRAL, VERDICT_HURT) for v in recent[-2:]))
        return {
            "intervention": intervention,
            "count": len(rows),
            "helped": helped,
            "neutral": neutral,
            "hurt": hurt,
            "last_verdict": last,
            "allowed_default": allowed,
        }

    def may_adjust_metric(self, intervention: str) -> bool:
        """Only evidence-supported improvements may move self-metrics."""
        s = self.intervention_summary(intervention)
        return s["helped"] > 0 and s["helped"] >= s["hurt"]

    def evidence_based_value(
        self,
        current: float,
        measured: float,
        intervention: str,
        max_step: float = 0.15,
    ) -> tuple[float, str]:
        """Return new metric value + rationale. Never free lunch."""
        if not self.may_adjust_metric(intervention):
            return float(current), f"no_evidence_{intervention}_must_not_move_metric"
        if measured >= current:
            return float(current), f"measured_{measured:.4f}_not_better_than_{current:.4f}"
        # Evidence-supported improvement: move toward measured, bounded step
        target = float(measured)
        if current - target > max_step:
            new_val = current - max_step
        else:
            new_val = target
        return max(0.0, new_val), f"evidence_move_{intervention}_{current:.4f}->{new_val:.4f}"

    @property
    def trials(self) -> list[LoopTrial]:
        return list(self._trials)

    def report(self) -> dict:
        by_int: dict[str, dict] = {}
        for t in self._trials:
            by_int.setdefault(t.intervention, self.intervention_summary(t.intervention))
        return {
            "trial_count": len(self._trials),
            "by_intervention": by_int,
            "recent": [t.to_dict() for t in self._trials[-10:]],
        }
