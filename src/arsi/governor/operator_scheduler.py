"""Operator Scheduler — MetaRSI-inspired two-axis scheduling.

Horizontal axis: which improvement operators to apply, in what order
Vertical axis: rewrite each operator's proposal policy

Based on: MetaRSI (arXiv:2609.06396)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from arsi.foundation.schema import EmpowermentDimension

logger = logging.getLogger(__name__)


@dataclass
class ImprovementOperator:
    """A schedulable improvement operator (one empowerment dimension)."""
    dimension: EmpowermentDimension
    cost: float = 1.0
    expected_gain: float = 0.5
    last_used_generation: int = 0
    signal_freshness: float = 1.0
    total_applications: int = 0
    successful_applications: int = 0
    proposal_policy_version: int = 1

    @property
    def success_rate(self) -> float:
        return self.successful_applications / max(self.total_applications, 1)

    @property
    def cost_benefit_ratio(self) -> float:
        return self.expected_gain / max(self.cost, 0.01)

    @property
    def is_stale(self) -> bool:
        return self.signal_freshness < 0.3


class OperatorScheduler:
    """MetaRSI-inspired two-axis scheduler with uncovered-dimension priority."""

    def __init__(self):
        self.operators: dict[EmpowermentDimension, ImprovementOperator] = {}
        self._init_default_operators()
        self._schedule_history: list[dict] = []
        self._generation = 0

    def _init_default_operators(self):
        defaults = {
            EmpowermentDimension.KNOWLEDGE: ImprovementOperator(
                dimension=EmpowermentDimension.KNOWLEDGE, cost=2.0, expected_gain=0.6,
            ),
            EmpowermentDimension.DECOMPOSITION: ImprovementOperator(
                dimension=EmpowermentDimension.DECOMPOSITION, cost=1.5, expected_gain=0.5,
            ),
            EmpowermentDimension.CALIBRATION: ImprovementOperator(
                dimension=EmpowermentDimension.CALIBRATION, cost=1.0, expected_gain=0.4,
            ),
            EmpowermentDimension.ATTENTION: ImprovementOperator(
                dimension=EmpowermentDimension.ATTENTION, cost=1.0, expected_gain=0.45,
            ),
            EmpowermentDimension.METACOGNITION: ImprovementOperator(
                dimension=EmpowermentDimension.METACOGNITION, cost=1.2, expected_gain=0.5,
            ),
            EmpowermentDimension.ENVIRONMENT: ImprovementOperator(
                dimension=EmpowermentDimension.ENVIRONMENT, cost=1.0, expected_gain=0.4,
            ),
        }
        # Keep only implemented set (legacy tests expect 6); uncovered may name others
        self.operators = defaults

    def revise_proposal_policy(self, dimension: EmpowermentDimension) -> dict:
        """MetaRSI vertical axis: revise an operator's proposal policy."""
        if dimension not in self.operators:
            return {"status": "unknown_dimension"}
        op = self.operators[dimension]
        if op.success_rate < 0.3 and op.total_applications >= 3:
            op.proposal_policy_version += 1
            op.cost *= 0.8
            return {
                "status": "revised",
                "dimension": dimension.value,
                "new_version": op.proposal_policy_version,
                "reason": f"low_success_rate:{op.success_rate:.2f}",
            }
        return {"status": "no_revision_needed"}

    def schedule(
        self,
        budget: float = 3.0,
        max_operators: int = 3,
        uncovered: Optional[list] = None,
    ) -> list[ImprovementOperator]:
        """Select operators under budget; force-cover uncovered dimensions first."""
        self._generation += 1
        for op in self.operators.values():
            op.signal_freshness = max(0.0, op.signal_freshness - 0.05)

        selected: list[ImprovementOperator] = []
        used = 0.0
        forced = []
        if uncovered:
            for u in uncovered:
                dim = None
                if isinstance(u, EmpowermentDimension):
                    dim = u
                else:
                    try:
                        dim = EmpowermentDimension(u)
                    except Exception:
                        continue
                if dim in self.operators and self.operators[dim] not in forced:
                    forced.append(self.operators[dim])
        for op in forced:
            if len(selected) >= max_operators:
                break
            if used + op.cost <= budget * 1.25:
                selected.append(op)
                used += op.cost

        remaining = [op for op in self.operators.values() if op not in selected and not op.is_stale]
        remaining.sort(key=lambda o: o.cost_benefit_ratio, reverse=True)
        for op in remaining:
            if len(selected) >= max_operators:
                break
            if used + op.cost <= budget:
                selected.append(op)
                used += op.cost
        # all stale + no uncovered → empty schedule (MetaRSI: no blind work)
        if not selected and not forced:
            self._schedule_history.append({
                "generation": self._generation,
                "selected": [],
                "uncovered_forced": [],
                "budget": budget,
                "used": 0,
                "timestamp": datetime.now().isoformat(),
            })
            logger.info(f"Schedule gen={self._generation}: [] (all_stale)")
            return []
        if not selected:
            selected = list(self.operators.values())[:max_operators]
        for op in selected:
            op.total_applications += 1
            op.signal_freshness = min(1.0, op.signal_freshness + 0.2)
            op.last_used_generation = self._generation
        self._schedule_history.append({
            "generation": self._generation,
            "selected": [op.dimension.value for op in selected],
            "uncovered_forced": [op.dimension.value for op in forced if op in selected],
            "budget": budget,
            "used": used,
            "timestamp": datetime.now().isoformat(),
        })
        logger.info(
            f"Schedule gen={self._generation}: {[op.dimension.value for op in selected]} "
            f"forced={[op.dimension.value for op in forced if op in selected]}"
        )
        return selected

    def mark_capability_change(self):
        for op in self.operators.values():
            op.signal_freshness *= 0.5

    def record_application(self, dimension: EmpowermentDimension, success: bool):
        if dimension in self.operators:
            op = self.operators[dimension]
            op.total_applications += 1
            if success:
                op.successful_applications += 1
            op.last_used_generation = self._generation
            op.expected_gain = op.success_rate

    @property
    def stats(self) -> dict:
        return {
            "generation": self._generation,
            "operators": {
                d.value: {
                    "cost": o.cost,
                    "gain": o.expected_gain,
                    "apps": o.total_applications,
                    "success_rate": o.success_rate,
                    "freshness": o.signal_freshness,
                }
                for d, o in self.operators.items()
            },
            "history_tail": self._schedule_history[-5:],
        }
