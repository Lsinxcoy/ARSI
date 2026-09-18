"""TransitionLedger — self-prediction vs reality, auditable.

Q4 support: what actually happened to "me" after action X.
No claim without a record.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional


@dataclass
class TransitionRecord:
    timestamp: str = ""
    organ: str = ""
    action: str = ""
    predicted: object = None
    actual: object = None
    error: float = 0.0
    correct: Optional[bool] = None
    evidence_refs: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)


class TransitionLedger:
    """Append-only ledger of self-model transitions."""

    def __init__(self, path: Optional[str | Path] = None, window: int = 50):
        self.window = window
        self._recent: Deque[TransitionRecord] = deque(maxlen=200)
        self._by_organ: dict[str, Deque[TransitionRecord]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self.path = Path(path) if path else None
        self._seq = 0

    def append(self, record: TransitionRecord) -> TransitionRecord:
        self._seq += 1
        self._recent.append(record)
        self._by_organ[record.organ].append(record)
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            except Exception:
                pass
        return record

    def record_prediction(
        self,
        organ: str,
        action: str,
        predicted,
        actual,
        error: Optional[float] = None,
        correct: Optional[bool] = None,
        evidence_refs: Optional[list[str]] = None,
        meta: Optional[dict] = None,
    ) -> TransitionRecord:
        if error is None:
            if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
                error = abs(float(predicted) - float(actual))
            elif correct is not None:
                error = 0.0 if correct else 1.0
            else:
                error = 0.0 if predicted == actual else 1.0
        if correct is None:
            correct = error < 0.5
        return self.append(TransitionRecord(
            organ=organ,
            action=action,
            predicted=predicted,
            actual=actual,
            error=float(error),
            correct=bool(correct),
            evidence_refs=list(evidence_refs or []),
            meta=dict(meta or {}),
        ))

    def recent(self, n: int = 20, organ: Optional[str] = None) -> list[dict]:
        rows = self._by_organ[organ] if organ else self._recent
        items = list(rows)[-n:]
        return [r.to_dict() for r in items]

    def rolling_error(self, organ: str, n: int = 20) -> float:
        rows = list(self._by_organ.get(organ, []))[-n:]
        if not rows:
            return 1.0
        return sum(r.error for r in rows) / len(rows)

    def rolling_accuracy(self, organ: str, n: int = 20) -> float:
        rows = list(self._by_organ.get(organ, []))[-n:]
        if not rows:
            return 0.0
        return sum(1 for r in rows if r.correct) / len(rows)

    @property
    def size(self) -> int:
        return len(self._recent)

    def stats(self) -> dict:
        return {
            "size": self.size,
            "organs": {k: len(v) for k, v in self._by_organ.items()},
            "path": str(self.path) if self.path else None,
        }
