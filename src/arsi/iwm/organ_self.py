"""OrganSelfModel — per-organ reliability with control hooks.

Reliability is rolling evidence, not a constant. Status=unknown organs
must not drive control decisions.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Optional


ORGAN_BEHAVIOR_PREDICTOR = "behavior_predictor"
ORGAN_DYNAMICS = "dynamics"
ORGAN_DREAM = "dream"
ORGAN_PORTFOLIO = "portfolio"
ORGAN_MEMORY = "memory"

KNOWN_ORGANS = (
    ORGAN_BEHAVIOR_PREDICTOR,
    ORGAN_DYNAMICS,
    ORGAN_DREAM,
    ORGAN_PORTFOLIO,
    ORGAN_MEMORY,
)

STATUS_OK = "ok"
STATUS_UNRELIABLE = "unreliable"
STATUS_UNKNOWN = "unknown"


@dataclass
class OrganReliability:
    organ: str
    samples: int = 0
    hits: int = 0
    misses: int = 0
    window: Deque[int] = field(default_factory=lambda: deque(maxlen=30))
    status: str = STATUS_UNKNOWN
    reliability: float = 0.0
    control_hint: str = ""
    last_updated: str = ""

    def to_dict(self) -> dict:
        return {
            "organ": self.organ,
            "samples": self.samples,
            "hits": self.hits,
            "misses": self.misses,
            "reliability": round(self.reliability, 4),
            "status": self.status,
            "control_hint": self.control_hint,
            "last_updated": self.last_updated,
        }


class OrganSelfModel:
    """Q1: which organs are unreliable, and control must change because of it."""

    def __init__(self, min_samples: int = 3, unreliable_below: float = 0.35):
        self.min_samples = min_samples
        self.unreliable_below = unreliable_below
        self._organs: dict[str, OrganReliability] = {
            name: OrganReliability(organ=name) for name in KNOWN_ORGANS
        }

    def _get(self, organ: str) -> OrganReliability:
        if organ not in self._organs:
            self._organs[organ] = OrganReliability(organ=organ)
        return self._organs[organ]

    def record(self, organ: str, correct: bool, control_hint: str = "") -> OrganReliability:
        rec = self._get(organ)
        rec.samples += 1
        if correct:
            rec.hits += 1
        else:
            rec.misses += 1
        rec.window.append(1 if correct else 0)
        rec.last_updated = datetime.now().isoformat()
        if control_hint:
            rec.control_hint = control_hint
        self._recompute(rec)
        return rec

    def record_score(self, organ: str, score: float, control_hint: str = "") -> OrganReliability:
        """Treat continuous score in [0,1] as soft evidence (>=0.5 counts as hit-ish)."""
        score = max(0.0, min(1.0, float(score)))
        rec = self._get(organ)
        rec.samples += 1
        if score >= 0.5:
            rec.hits += 1
        else:
            rec.misses += 1
        # soft window: store rounded score bucket
        rec.window.append(1 if score >= 0.5 else 0)
        rec.last_updated = datetime.now().isoformat()
        if control_hint:
            rec.control_hint = control_hint
        # blend reliability toward observed score for smoother control
        if rec.reliability <= 0.0 and rec.samples == 1:
            rec.reliability = score
        self._recompute(rec, override_value=score if rec.samples >= self.min_samples else None)
        return rec

    def _recompute(self, rec: OrganReliability, override_value: Optional[float] = None) -> None:
        if rec.samples < self.min_samples:
            rec.status = STATUS_UNKNOWN
            rec.reliability = 0.0 if rec.samples == 0 else rec.hits / rec.samples
            return
        if override_value is not None:
            # exponential blend toward latest soft score
            rec.reliability = 0.7 * rec.reliability + 0.3 * override_value
        else:
            rec.reliability = sum(rec.window) / max(len(rec.window), 1)
        if rec.reliability < self.unreliable_below:
            rec.status = STATUS_UNRELIABLE
        else:
            rec.status = STATUS_OK

    def reliability(self, organ: str) -> float:
        return self._get(organ).reliability

    def status(self, organ: str) -> str:
        return self._get(organ).status

    def trust_weight(self, organ: str) -> float:
        """Control weight: unknown/unreliable organs do not drive behavior."""
        st = self.status(organ)
        if st == STATUS_UNKNOWN:
            return 0.0
        if st == STATUS_UNRELIABLE:
            return 0.0
        return max(0.1, min(1.0, self.reliability(organ)))

    def unreliable_organs(self) -> list[str]:
        return [n for n, r in self._organs.items() if r.status == STATUS_UNRELIABLE]

    def control_hooks(self) -> dict:
        """Behavioral consequences — without hooks this is only a dashboard."""
        mem_status = self.status(ORGAN_MEMORY)
        mem_trust = self.trust_weight(ORGAN_MEMORY)
        mem_ok = mem_status == STATUS_OK and mem_trust >= 0.5
        mem_bad = mem_status == STATUS_UNRELIABLE or mem_trust <= 0.0
        hooks = {
            "downweight_pre_enactment": self.trust_weight(ORGAN_DYNAMICS) <= 0.0,
            "forbid_default_dream": self.status(ORGAN_DREAM) == STATUS_UNRELIABLE,
            "downweight_portfolio": self.trust_weight(ORGAN_PORTFOLIO) <= 0.0,
            "prefer_learn_over_evolve": self.trust_weight(ORGAN_BEHAVIOR_PREDICTOR) <= 0.0,
            "memory_organ_unreliable": mem_bad,
            "memory_organ_trusted": mem_ok,
            "trust_memory_for_learn": mem_ok,
            "downweight_memory_ops": mem_bad,
            "prefer_remember_ingest": mem_bad,
            "memory_trust": mem_trust,
            "memory_status": mem_status,
            "unreliable_organs": self.unreliable_organs(),
        }
        return hooks

    def snapshot(self) -> dict:
        return {
            name: rec.to_dict() for name, rec in sorted(self._organs.items())
        }

    def report(self) -> dict:
        return {
            "organs": self.snapshot(),
            "hooks": self.control_hooks(),
            "unreliable": self.unreliable_organs(),
        }
