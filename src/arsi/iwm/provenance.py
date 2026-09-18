"""DecisionProvenance — Q5: why was action a chosen?

Replayable decision records with IWM snapshot.
"""
from __future__ import annotations

import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Deque, Optional


@dataclass
class DecisionRecord:
    decision_id: str = ""
    action: str = ""
    reason: str = ""
    decision_source: str = ""
    candidates: list[str] = field(default_factory=list)
    selected_score: Optional[float] = None
    state_digest: dict = field(default_factory=dict)
    iwm_snapshot: dict = field(default_factory=dict)
    hooks_applied: list[str] = field(default_factory=list)
    confidence: float = 0.0
    outcome: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.decision_id:
            self.decision_id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionProvenance:
    def __init__(self, capacity: int = 200):
        self._records: Deque[DecisionRecord] = deque(maxlen=capacity)
        self._by_id: dict[str, DecisionRecord] = {}

    def record(
        self,
        action: str,
        reason: str = "",
        decision_source: str = "",
        candidates: Optional[list[str]] = None,
        state_digest: Optional[dict] = None,
        iwm_snapshot: Optional[dict] = None,
        hooks_applied: Optional[list[str]] = None,
        confidence: float = 0.0,
        selected_score: Optional[float] = None,
    ) -> DecisionRecord:
        rec = DecisionRecord(
            action=action,
            reason=reason,
            decision_source=decision_source,
            candidates=list(candidates or []),
            state_digest=dict(state_digest or {}),
            iwm_snapshot=dict(iwm_snapshot or {}),
            hooks_applied=list(hooks_applied or []),
            confidence=float(confidence),
            selected_score=selected_score,
        )
        self._records.append(rec)
        self._by_id[rec.decision_id] = rec
        return rec

    def mark_outcome(self, decision_id: str, outcome: str) -> Optional[DecisionRecord]:
        rec = self._by_id.get(decision_id)
        if rec:
            rec.outcome = outcome
        return rec

    def replay(self, decision_id: str) -> Optional[dict]:
        rec = self._by_id.get(decision_id)
        return rec.to_dict() if rec else None

    def recent(self, n: int = 10) -> list[dict]:
        return [r.to_dict() for r in list(self._records)[-n:]]

    @property
    def count(self) -> int:
        return len(self._records)

    def stats(self) -> dict:
        return {"count": self.count, "last_action": self._records[-1].action if self._records else None}
