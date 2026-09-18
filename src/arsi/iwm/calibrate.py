"""Introspector self-check (Q6) — is the introspector itself trustworthy?

Low self_trust → IWM control advice is ignored; system degrades to baseline.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque


@dataclass
class ControlOutcome:
    decision_id: str
    used_iwm: bool
    success: bool
    note: str = ""
    timestamp: str = ""


class IntrospectorCalibrator:
    def __init__(self, min_samples: int = 5, degrade_below: float = 0.35):
        self.min_samples = min_samples
        self.degrade_below = degrade_below
        self._outcomes: Deque[ControlOutcome] = deque(maxlen=100)
        self._iwm_hits = 0
        self._iwm_total = 0
        self._baseline_hits = 0
        self._baseline_total = 0

    def record(self, decision_id: str, used_iwm: bool, success: bool, note: str = "") -> None:
        ts = datetime.now().isoformat()
        self._outcomes.append(ControlOutcome(decision_id, used_iwm, success, note, ts))
        if used_iwm:
            self._iwm_total += 1
            if success:
                self._iwm_hits += 1
        else:
            self._baseline_total += 1
            if success:
                self._baseline_hits += 1

    @property
    def iwm_success_rate(self) -> float:
        if self._iwm_total == 0:
            return 0.0
        return self._iwm_hits / self._iwm_total

    @property
    def baseline_success_rate(self) -> float:
        if self._baseline_total == 0:
            return 0.0
        return self._baseline_hits / self._baseline_total

    def self_trust(self) -> float:
        """Trust in IWM-based control. Unknown until enough IWM outcomes."""
        if self._iwm_total < self.min_samples:
            return 0.5  # neutral unknown — do not fully ignore, do not fully trust
        return self.iwm_success_rate

    def should_degrade_to_baseline(self) -> bool:
        if self._iwm_total < self.min_samples:
            return False
        if self.iwm_success_rate < self.degrade_below:
            return True
        # IWM clearly worse than baseline with enough samples
        if (
            self._baseline_total >= self.min_samples
            and self.iwm_success_rate + 0.1 < self.baseline_success_rate
        ):
            return True
        return False

    def report(self) -> dict:
        return {
            "iwm_total": self._iwm_total,
            "iwm_success_rate": round(self.iwm_success_rate, 4),
            "baseline_total": self._baseline_total,
            "baseline_success_rate": round(self.baseline_success_rate, 4),
            "self_trust": round(self.self_trust(), 4),
            "degrade_to_baseline": self.should_degrade_to_baseline(),
            "min_samples": self.min_samples,
            "degrade_below": self.degrade_below,
        }
