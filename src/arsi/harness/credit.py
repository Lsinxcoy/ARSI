"""RRSI evidence-aware credit assignment L_t (arXiv:2609.24972 App.C.2 Eq.10–11).

Each atomic edit record: (t, component, hypothesis, diff, dS, dC, accepted).
Rejected hypotheses stay negative evidence; winners keep explicit credit.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EditRecord:
    t: int
    component: str
    hypothesis: str = ""
    diff: str = ""
    dS: float = 0.0
    dC: float = 0.0
    accepted: bool = False
    candidate_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class CreditLedger:
    """L_t + component summaries T_t, g_t(ℓ), N_t(ℓ), prune set B_t."""

    def __init__(self, prune_window: int = 4):
        self.prune_window = max(1, int(prune_window))
        self.records: list[EditRecord] = []

    def log_candidate(
        self,
        t: int,
        candidate_id: str,
        edits: list[dict],
        dS: float,
        dC: float,
        accepted: bool,
    ) -> None:
        """One history row per atomic edit; shared ΔS/ΔC/outcome (paper App.C.2)."""
        for e in edits or []:
            self.records.append(
                EditRecord(
                    t=int(t),
                    component=str((e or {}).get("component") or (e or {}).get("dimension") or "prompt"),
                    hypothesis=str((e or {}).get("hypothesis") or (e or {}).get("summary") or ""),
                    diff=str((e or {}).get("diff") or ""),
                    dS=float(dS),
                    dC=float(dC),
                    accepted=bool(accepted),
                    candidate_id=str(candidate_id),
                )
            )

    def exercised(self) -> set[str]:
        return {r.component for r in self.records}

    def accepted_count(self, component: str) -> int:
        """N_t(ℓ) — accepted edits tagged with ℓ before now."""
        return sum(1 for r in self.records if r.component == component and r.accepted)

    def recent_best_gain(self, component: str, t_now: Optional[int] = None) -> float:
        """g_t(ℓ) = max ΔS over pruning window; empty → -inf."""
        t_now = self.records[-1].t if t_now is None and self.records else (t_now or 0)
        vals = [
            r.dS
            for r in self.records
            if r.component == component and (t_now - r.t) <= self.prune_window
        ]
        return max(vals) if vals else float("-inf")

    def prune_targets(self) -> list[str]:
        """B_t = {ℓ exercised, g_t(ℓ) ≤ 0}"""
        out = []
        for c in self.exercised():
            g = self.recent_best_gain(c)
            if g <= 0:
                out.append(c)
        return sorted(out)

    def rejected_hypotheses(self) -> list[str]:
        """Negative evidence for the proposer."""
        seen = []
        for r in self.records:
            if (not r.accepted) and r.hypothesis and r.hypothesis not in seen:
                seen.append(r.hypothesis)
        return seen

    def to_dict(self) -> dict:
        return {
            "prune_window": self.prune_window,
            "n": len(self.records),
            "exercised": sorted(self.exercised()),
            "prune_targets": self.prune_targets(),
            "rejected_hypotheses": self.rejected_hypotheses()[:50],
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps([r.to_dict() for r in self.records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
