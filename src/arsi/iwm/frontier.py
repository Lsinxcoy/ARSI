"""KnowledgeFrontier — Q2: what do I know vs not know.

Frontier must change explore/exploit, not just sit in a report.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FrontierCell:
    key: str
    support: int = 0
    successes: int = 0
    failures: int = 0
    mean_effect: float = 0.0
    _effects: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.support <= 0:
            return 0.0
        return self.successes / self.support

    @property
    def known(self) -> bool:
        return self.support >= 5 and self.success_rate >= 0.5

    @property
    def underexplored(self) -> bool:
        return self.support < 3

    @property
    def weak(self) -> bool:
        return self.support >= 3 and self.success_rate < 0.4

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "support": self.support,
            "success_rate": round(self.success_rate, 4),
            "mean_effect": round(self.mean_effect, 4),
            "known": self.known,
            "underexplored": self.underexplored,
            "weak": self.weak,
        }


class KnowledgeFrontier:
    def __init__(self):
        self._cells: dict[str, FrontierCell] = {}
        self._dimension_cells: dict[str, FrontierCell] = {}

    @staticmethod
    def _key(action: str) -> str:
        from arsi.world_model.siwm import BehaviorPredictor
        return BehaviorPredictor.categorize_action(action)

    def observe(self, action: str, outcome: str = "", effect: float = 0.0, dimension: str = "") -> None:
        key = self._key(action)
        cell = self._cells.get(key)
        if cell is None:
            cell = FrontierCell(key=key)
            self._cells[key] = cell
        cell.support += 1
        success = "success" in (outcome or "").lower() or float(effect) >= 0.6
        if success:
            cell.successes += 1
        else:
            cell.failures += 1
        cell._effects.append(float(effect))
        cell.mean_effect = sum(cell._effects[-20:]) / len(cell._effects[-20:])

        if dimension:
            dcell = self._dimension_cells.get(dimension)
            if dcell is None:
                dcell = FrontierCell(key=dimension)
                self._dimension_cells[dimension] = dcell
            dcell.support += 1
            if success:
                dcell.successes += 1
            else:
                dcell.failures += 1

    def query(self, key: str) -> dict:
        cell = self._cells.get(key) or self._cells.get(self._key(key))
        return cell.to_dict() if cell else {"key": key, "support": 0, "known": False}

    def frontier(self) -> dict:
        known = [c.to_dict() for c in self._cells.values() if c.known]
        under = [c.to_dict() for c in self._cells.values() if c.underexplored]
        weak = [c.to_dict() for c in self._cells.values() if c.weak]
        return {
            "known": known,
            "underexplored": under,
            "weak": weak,
            "dimensions": {k: v.to_dict() for k, v in self._dimension_cells.items()},
            "updated_at": datetime.now().isoformat(),
        }

    def explore_bias(self) -> list[str]:
        """Action categories that should get more exploration weight."""
        bias = []
        for cell in self._cells.values():
            if cell.underexplored or cell.weak:
                bias.append(cell.key)
        # Prefer weak (seen but failing) then underexplored
        weak = [c.key for c in self._cells.values() if c.weak]
        under = [c.key for c in self._cells.values() if c.underexplored]
        return weak + [k for k in under if k not in weak]

    def exploit_bias(self) -> list[str]:
        return [c.key for c in self._cells.values() if c.known]

    def should_explore(self, action: str) -> bool:
        key = self._key(action)
        cell = self._cells.get(key)
        if cell is None:
            return True
        return cell.underexplored or cell.weak

    def report(self) -> dict:
        fr = self.frontier()
        return {
            **fr,
            "explore_bias": self.explore_bias(),
            "exploit_bias": self.exploit_bias(),
        }
