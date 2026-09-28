"""Recuris R0 — Working Memory + evidence checkers (arXiv:2608.24876).

WM tracks goals with verified status. Agents may *propose* updates; only
environment evidence lets the checker *commit* done. Verbal completion is
rejected (paper: truth_bounce / executable request NOT EXECUTED).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_BLOCKED = "blocked"


@dataclass
class GoalEntry:
    goal_id: str
    content: str
    status: str = STATUS_PENDING
    evidence: list = field(default_factory=list)
    blocker: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StateUpdate:
    goal_id: str
    new_status: str
    evidence: list = field(default_factory=list)
    blocker: str = ""
    claimed: bool = False  # agent assertion without env evidence

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CheckerVerdict:
    accepted: bool
    reason: str
    update: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


def checker_evidence_required(update: StateUpdate) -> CheckerVerdict:
    """Recuris: done requires environment evidence; claim-only is bounced."""
    st = str(update.new_status or "")
    if st == STATUS_DONE:
        ev = [e for e in (update.evidence or []) if e]
        if not ev or update.claimed and not ev:
            return CheckerVerdict(False, "truth_bounce_no_env_evidence", update.to_dict())
        if not ev:
            return CheckerVerdict(False, "done_requires_evidence", update.to_dict())
    if st == STATUS_BLOCKED and not (update.blocker or "").strip():
        return CheckerVerdict(False, "blocked_requires_blocker", update.to_dict())
    return CheckerVerdict(True, "ok", update.to_dict())


def checker_env_says_done(update: StateUpdate, env_signals: Sequence[str]) -> CheckerVerdict:
    """Optional strict mode: done only if an env signal matches evidence id."""
    if str(update.new_status) != STATUS_DONE:
        return checker_evidence_required(update)
    sigs = set(str(s) for s in (env_signals or []))
    ev = [str(e) for e in (update.evidence or []) if e]
    if not any(e in sigs for e in ev):
        return CheckerVerdict(False, "env_signal_missing_for_done", update.to_dict())
    return CheckerVerdict(True, "env_confirmed", update.to_dict())


class WorkingState:
    """Structured working state instance (W spec)."""

    def __init__(self, goals: Optional[Sequence[GoalEntry]] = None, task_id: str = ""):
        self.task_id = task_id
        self.goals: dict[str, GoalEntry] = {}
        for g in goals or []:
            self.goals[g.goal_id] = g
        self._log: list[dict] = []

    @classmethod
    def init_from_task(cls, task_description: str, goal_texts: Sequence[str], task_id: str = "") -> "WorkingState":
        goals = []
        for i, text in enumerate(goal_texts or []):
            goals.append(GoalEntry(goal_id=f"g{i}", content=str(text), status=STATUS_PENDING))
        if not goals:
            goals = [GoalEntry(goal_id="g0", content=str(task_description or "task"), status=STATUS_PENDING)]
        return cls(goals, task_id=task_id or "task")

    def propose_update(self, update: StateUpdate) -> dict:
        """Run checkers; commit only if accepted (Recuris verified EM–WM)."""
        v = checker_evidence_required(update)
        row = v.to_dict()
        row["task_id"] = self.task_id
        self._log.append(row)
        if not v.accepted:
            return row
        g = self.goals.get(update.goal_id)
        if g is None:
            g = GoalEntry(goal_id=update.goal_id, content=update.goal_id)
            self.goals[update.goal_id] = g
        g.status = str(update.new_status)
        g.evidence = list(update.evidence or []) + list(g.evidence or [])
        g.blocker = update.blocker or g.blocker
        return row

    def unresolved(self) -> list[str]:
        return [g.goal_id for g in self.goals.values() if g.status != STATUS_DONE]

    def snapshot(self) -> dict:
        return {
            "task_id": self.task_id,
            "goals": [g.to_dict() for g in self.goals.values()],
            "unresolved": self.unresolved(),
            "n_log": len(self._log),
        }

    def to_dict(self) -> dict:
        return self.snapshot()
