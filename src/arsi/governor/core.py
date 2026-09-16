"""Autopoietic Governor — the system's decision-making organ.

Reads joint world state S_t = (Φ_t, Ψ_t), outputs coupled action a = (a_phy, a_ment).
Governor's own strategy parameters φ are distilled from the effect ledger.
Constrained by iron laws — can evolve but cannot rewrite them.

Based on: ARSI Whitepaper v0.8 §6 + §7
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from arsi.foundation.iron_laws import IronLaws
from arsi.foundation.schema import (
    CoupledAction,
    EmpowermentDimension,
    IntentRecord,
    WorldState,
)
from arsi.world_model.siwm import SIWM

logger = logging.getLogger(__name__)


class DimensionStatus:
    """Status of a single empowerment dimension."""

    def __init__(
        self,
        dimension: EmpowermentDimension,
        stage: str = "infancy",
        marginal_gain: float = 0.5,
        data_sufficiency: float = 0.0,
        last_evolved: Optional[datetime] = None,
    ):
        self.dimension = dimension
        self.stage = stage
        self.marginal_gain = marginal_gain
        self.data_sufficiency = data_sufficiency
        self.last_evolved = last_evolved


class DimensionManager:
    """Manages lifecycle of nine empowerment dimensions.

    Key design: each term focuses on 2-3 dimensions (rotation, not all-at-once).
    """

    STAGE_WEIGHTS = {
        "infancy": 0.3,
        "growth": 1.0,
        "maturity": 0.5,
        "decline": 0.8,
    }

    def __init__(self, max_per_term: int = 2, min_data_sufficiency: float = 0.3):
        self.max_per_term = max_per_term
        self.min_data = min_data_sufficiency
        self._history: dict[EmpowermentDimension, list[str]] = {
            d: [] for d in EmpowermentDimension
        }

    def scan(self, state: WorldState) -> dict[EmpowermentDimension, DimensionStatus]:
        """Scan all nine dimensions' current status."""
        trace_count = state.phi.storage_stats.get("trace_count", 0)
        base_sufficiency = min(1.0, trace_count / 100.0)

        status = {}
        for dim in EmpowermentDimension:
            # Estimate stage from history
            history = self._history[dim]
            if len(history) == 0:
                stage = "infancy"
            elif len(history) < 3:
                stage = "growth"
            elif len(history) < 10:
                stage = "maturity"
            else:
                stage = "decline" if len(history) > 20 else "maturity"

            status[dim] = DimensionStatus(
                dimension=dim,
                stage=stage,
                marginal_gain=self._estimate_gain(dim, state),
                data_sufficiency=base_sufficiency,
            )
        return status

    def select_priority(
        self, status: dict[EmpowermentDimension, DimensionStatus], k: Optional[int] = None
    ) -> list[EmpowermentDimension]:
        """Select k dimensions to focus on this term."""
        k = k or self.max_per_term
        scored = []
        for dim, s in status.items():
            if s.stage == "infancy" and s.data_sufficiency < self.min_data:
                continue
            score = s.marginal_gain * s.data_sufficiency * self.STAGE_WEIGHTS.get(s.stage, 0.5)
            scored.append((dim, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [d for d, _ in scored[:k]]

        # Record selection
        for d in selected:
            self._history[d].append(datetime.now().isoformat())

        return selected

    def _estimate_gain(self, dim: EmpowermentDimension, state: WorldState) -> float:
        """Estimate marginal gain for a dimension (simplified)."""
        # Basic heuristic: dimensions with fewer historical evolutions have more headroom
        history_len = len(self._history[dim])
        return max(0.1, 1.0 / (1.0 + history_len * 0.3))


class AutopoieticGovernor:
    """The Governor — ARSI's decision-making organ.

    Invariants:
    - Reads S_t = (Φ_t, Ψ_t)
    - Outputs a = (a_phy, a_ment)
    - a_ment is recorded for auditability
    - Policy φ is distilled from effect ledger (metabolism)
    - Cannot violate iron laws
    """

    def __init__(
        self,
        siwm: SIWM,
        store,
        iron_laws: IronLaws,
        dim_manager: Optional[DimensionManager] = None,
    ):
        self.siwm = siwm
        self.store = store
        self.laws = iron_laws
        self.dims = dim_manager or DimensionManager()
        self.policy: dict = {}
        self._decision_count = 0

    def decide(self, state: WorldState) -> CoupledAction:
        """Core decision loop."""
        self._decision_count += 1

        # Step 1: Check iron laws
        if self.laws.violated(state):
            return CoupledAction.freeze("铁律触发，系统冻结")

        # Step 2: Check η (self-model freshness)
        if state.eta >= 0.40:
            return CoupledAction(
                a_phy="dream",
                a_ment=IntentRecord(
                    reason=f"η={state.eta:.2f} 超阈值，需刷新自我模型",
                    expected_effect="η 下降",
                    risk_assessment="low",
                ),
            )

        # Step 3: Dimension lifecycle scan
        dim_status = self.dims.scan(state)
        target_dims = self.dims.select_priority(dim_status)

        # Step 4: Generate candidates and select
        candidates = self._generate_candidates(state, target_dims)

        # Step 5: Select best (with or without pre-enactment)
        depth = self.siwm.eta.adaptive_depth()
        if depth > 0 and len(candidates) > 1:
            best = self._pre_enact_select(candidates, state, depth)
        else:
            best = self._heuristic_select(candidates)

        # Step 6: Output coupled action with introspection
        return CoupledAction(
            a_phy=best["action"],
            a_ment=IntentRecord(
                reason=best["reason"],
                belief_ref=best.get("belief_refs", []),
                expected_effect=best.get("expected", ""),
                risk_assessment=best.get("risk", "low"),
                timestamp=datetime.now(),
            ),
            target_agent=best.get("target_agent"),
            params=best.get("params", {}),
        )

    def _generate_candidates(
        self, state: WorldState, target_dims: list[EmpowermentDimension]
    ) -> list[dict]:
        """Generate candidate actions based on state and target dimensions."""
        candidates = []

        # Always consider dream if η is rising
        if state.eta >= 0.20:
            candidates.append({
                "action": "dream",
                "reason": f"η={state.eta:.2f}，预防性梦境刷新",
                "expected": "η 稳定或下降",
                "risk": "low",
            })

        # Consider learn if there are unprocessed traces
        trace_count = state.phi.storage_stats.get("trace_count", 0)
        exp_count = state.phi.storage_stats.get("experience_count", 0)
        if trace_count > exp_count * 3:
            candidates.append({
                "action": "learn",
                "reason": f"PROXY 区有 {trace_count} 条轨迹未蒸馏",
                "expected": "经验记忆增加",
                "risk": "low",
            })

        # Consider evolve if we have enough experience
        if exp_count > 10:
            candidates.append({
                "action": "evolve",
                "reason": f"经验记忆 {exp_count} 条，可尝试机制突变",
                "expected": "机制效果提升",
                "risk": "medium",
            })

        # Consider maintain if memory is growing
        if trace_count > 100:
            candidates.append({
                "action": "maintain",
                "reason": f"轨迹积累 {trace_count} 条，需要清理",
                "expected": "存储优化",
                "risk": "low",
            })

        # Default: remember (ingest new data)
        if not candidates:
            candidates.append({
                "action": "remember",
                "reason": "无特殊需求，执行常规摄入",
                "expected": "记忆增长",
                "risk": "low",
            })

        return candidates

    def _pre_enact_select(
        self, candidates: list[dict], state: WorldState, depth: int
    ) -> dict:
        """Select best candidate via simplified pre-enactment.

        Full implementation would use SIWM Layer 2/3.
        For now, uses heuristic scoring.
        """
        best = candidates[0]
        best_score = -1.0

        for c in candidates:
            score = self._score_candidate(c, state)
            if score > best_score:
                best_score = score
                best = c

        return best

    def _heuristic_select(self, candidates: list[dict]) -> dict:
        """Heuristic selection when pre-enactment is unavailable."""
        priority = {"dream": 3, "maintain": 2, "learn": 2, "evolve": 1, "remember": 1}
        return max(candidates, key=lambda c: priority.get(c["action"], 0))

    def _score_candidate(self, candidate: dict, state: WorldState) -> float:
        """Score a candidate action (simplified)."""
        base_scores = {
            "dream": 0.8 if state.eta > 0.25 else 0.3,
            "learn": 0.7,
            "evolve": 0.6,
            "maintain": 0.5,
            "remember": 0.4,
        }
        return base_scores.get(candidate["action"], 0.3)

    def distill_policy(self) -> dict:
        """Distill policy parameters from effect ledger (Governor's metabolism)."""
        ledger = self.store.get_effect_ledger(limit=100)
        if not ledger:
            return self.policy

        # Compute per-mechanism average effect
        mech_effects: dict[str, list[float]] = {}
        for entry in ledger:
            mech = entry.get("mechanism", "unknown")
            effect = entry.get("effect", 0.0)
            mech_effects.setdefault(mech, []).append(effect)

        self.policy = {
            "mechanism_preferences": {
                mech: sum(effects) / len(effects)
                for mech, effects in mech_effects.items()
            },
            "budget_allocation": self._compute_budget_allocation(mech_effects),
            "distilled_at": datetime.now().isoformat(),
        }

        # Validate against iron laws
        self.laws.validate_policy(self.policy)
        return self.policy

    def _compute_budget_allocation(self, mech_effects: dict) -> dict:
        """Compute budget allocation across dimensions."""
        dims = list(EmpowermentDimension)
        total = len(dims)
        base = 1.0 / total
        return {d.value: base for d in dims}

    @property
    def stats(self) -> dict:
        return {
            "decision_count": self._decision_count,
            "policy_keys": list(self.policy.keys()),
            "eta": self.siwm.eta.value,
        }
