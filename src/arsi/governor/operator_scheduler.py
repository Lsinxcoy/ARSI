"""Operator Scheduler — MetaRSI-inspired two-axis scheduling.

Horizontal axis: which improvement operators to apply, in what order
Vertical axis: rewrite each operator's proposal policy

Based on: MetaRSI (arXiv:2609.06396)
Key finding: scheduling itself is worth 3.6 points beyond the operators

MetaRSI Laws:
- Law 2: self-knowledge expires, re-description is the rate limit
- Law 3: capability is substrate-free but cost is not
- Law 5: loops don't create capability, all gains are imported
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from arsi.foundation.schema import EmpowermentDimension

logger = logging.getLogger(__name__)


@dataclass
class ImprovementOperator:
    """A schedulable improvement operator (one empowerment dimension)."""
    dimension: EmpowermentDimension
    cost: float = 1.0                    # estimated cost (LLM calls, time)
    expected_gain: float = 0.5           # estimated benefit
    last_used_generation: int = 0
    signal_freshness: float = 1.0        # MetaRSI Law 2
    total_applications: int = 0
    successful_applications: int = 0
    proposal_policy_version: int = 1     # MetaRSI: vertical axis

    @property
    def success_rate(self) -> float:
        return self.successful_applications / max(self.total_applications, 1)

    @property
    def cost_benefit_ratio(self) -> float:
        return self.expected_gain / max(self.cost, 0.01)

    @property
    def is_stale(self) -> bool:
        """MetaRSI Law 2: signals expire after capability changes."""
        return self.signal_freshness < 0.3


class OperatorScheduler:
    """MetaRSI-inspired two-axis scheduler.

    Horizontal: selects which operators to run, in what order
    Vertical: revises each operator's proposal policy
    """

    def __init__(self):
        self.operators: dict[EmpowermentDimension, ImprovementOperator] = {}
        self._init_default_operators()
        self._schedule_history: list[dict] = []
        self._generation = 0

    def _init_default_operators(self):
        """Initialize operators for all 9 dimensions."""
        defaults = {
            EmpowermentDimension.KNOWLEDGE: ImprovementOperator(
                dimension=EmpowermentDimension.KNOWLEDGE,
                cost=2.0, expected_gain=0.6,
            ),
            EmpowermentDimension.DECOMPOSITION: ImprovementOperator(
                dimension=EmpowermentDimension.DECOMPOSITION,
                cost=1.5, expected_gain=0.5,
            ),
            EmpowermentDimension.CALIBRATION: ImprovementOperator(
                dimension=EmpowermentDimension.CALIBRATION,
                cost=1.0, expected_gain=0.4,
            ),
            EmpowermentDimension.ATTENTION: ImprovementOperator(
                dimension=EmpowermentDimension.ATTENTION,
                cost=1.0, expected_gain=0.4,
            ),
            EmpowermentDimension.METACOGNITION: ImprovementOperator(
                dimension=EmpowermentDimension.METACOGNITION,
                cost=1.5, expected_gain=0.7,
            ),
            EmpowermentDimension.ENVIRONMENT: ImprovementOperator(
                dimension=EmpowermentDimension.ENVIRONMENT,
                cost=1.0, expected_gain=0.5,
            ),
        }
        self.operators = defaults

    def schedule(self, budget: float = 5.0, max_operators: int = 3) -> list[ImprovementOperator]:
        """Select operators for this term (MetaRSI horizontal axis).

        Rules:
        1. Signal freshness filter (Law 2)
        2. Cost-benefit ranking
        3. Budget constraint
        4. Max operators per term
        """
        self._generation += 1

        # Rule 1: Filter stale operators
        fresh_ops = [op for op in self.operators.values() if not op.is_stale]

        # Rule 2: Rank by cost-benefit
        scored = sorted(fresh_ops, key=lambda op: op.cost_benefit_ratio, reverse=True)

        # Rule 3: Budget + max constraint
        selected = []
        remaining_budget = budget
        for op in scored:
            if len(selected) >= max_operators:
                break
            if op.cost <= remaining_budget:
                selected.append(op)
                remaining_budget -= op.cost

        # Record schedule
        self._schedule_history.append({
            "generation": self._generation,
            "selected": [op.dimension.value for op in selected],
            "budget_used": budget - remaining_budget,
            "budget_total": budget,
            "timestamp": datetime.now().isoformat(),
        })

        logger.info(f"Schedule gen={self._generation}: {[op.dimension.value for op in selected]}")
        return selected

    def mark_capability_change(self):
        """MetaRSI Law 2: capability change invalidates prior signals.

        Called after any capability-altering action (evolve, learn, dream).
        """
        for op in self.operators.values():
            op.signal_freshness *= 0.5
            logger.debug(f"  {op.dimension.value}: freshness → {op.signal_freshness:.2f}")

    def record_application(self, dimension: EmpowermentDimension, success: bool):
        """Record the outcome of an operator application."""
        if dimension in self.operators:
            op = self.operators[dimension]
            op.total_applications += 1
            if success:
                op.successful_applications += 1
            op.last_used_generation = self._generation
            # Update expected gain based on actual results
            op.expected_gain = op.success_rate

    def revise_proposal_policy(self, dimension: EmpowermentDimension) -> dict:
        """MetaRSI vertical axis: revise an operator's proposal policy.

        When an operator consistently fails, change HOW it proposes,
        not just WHAT it proposes.
        """
        if dimension not in self.operators:
            return {"status": "unknown_dimension"}

        op = self.operators[dimension]

        if op.success_rate < 0.3 and op.total_applications >= 3:
            # Low success rate → revise policy
            op.proposal_policy_version += 1
            op.cost *= 0.8  # Lower cost expectation
            logger.info(f"Revised {dimension.value} policy → v{op.proposal_policy_version}")
            return {
                "status": "revised",
                "dimension": dimension.value,
                "new_version": op.proposal_policy_version,
                "reason": f"low_success_rate:{op.success_rate:.2f}",
            }

        return {"status": "no_revision_needed"}

    @property
    def stats(self) -> dict:
        return {
            "generation": self._generation,
            "schedule_count": len(self._schedule_history),
            "operators": {
                dim.value: {
                    "success_rate": op.success_rate,
                    "freshness": op.signal_freshness,
                    "cost_benefit": op.cost_benefit_ratio,
                    "applications": op.total_applications,
                    "policy_version": op.proposal_policy_version,
                }
                for dim, op in self.operators.items()
            },
        }
