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
        iwm=None,
    ):
        self.siwm = siwm
        self.store = store
        self.laws = iron_laws
        self.dims = dim_manager or DimensionManager()
        self.iwm = iwm
        self.policy: dict = {}
        self._decision_count = 0
        self._last_provenance_id = ""
        self._last_iwm_advice: dict = {}

    def decide(self, state: WorldState) -> CoupledAction:
        """Core decision loop — consumes IWM organ trust when available."""
        self._decision_count += 1
        hooks_applied: list[str] = []
        advice = {}
        if self.iwm is not None:
            advice = self.iwm.governor_advice(state)
            self._last_iwm_advice = advice
            if advice.get("degrade_to_baseline"):
                hooks_applied.append("degrade_to_baseline")
            if advice.get("downweight_pre_enactment"):
                hooks_applied.append("downweight_pre_enactment")
            if advice.get("forbid_default_dream"):
                hooks_applied.append("forbid_default_dream")
            if advice.get("prefer_learn"):
                hooks_applied.append("prefer_learn")
            if advice.get("trust_memory_for_learn"):
                hooks_applied.append("trust_memory_for_learn")
            if advice.get("downweight_memory_ops"):
                hooks_applied.append("downweight_memory_ops")
            if advice.get("prefer_remember_ingest"):
                hooks_applied.append("prefer_remember_ingest")

        # Step 1: Check iron laws — IWM cannot rewrite iron laws
        if self.laws.violated(state):
            return CoupledAction.freeze("铁律触发，系统冻结")

        # Step 2: η high → dream, unless dream organ is untrusted
        forbid_dream = bool(advice.get("forbid_default_dream"))
        if state.eta >= 0.40:
            if forbid_dream:
                # Q1 hook: untrusted dream → prefer learn/reflect path, keep η
                return CoupledAction(
                    a_phy="learn",
                    a_ment=IntentRecord(
                        reason=(
                            f"η={state.eta:.2f} 但 dream 器官不可靠"
                            f"（{advice.get('forbid_dream_reason') or 'loop_trial'}），改走 learn"
                        ),
                        expected_effect="经验蒸馏，不做无证据 dream",
                        risk_assessment="low",
                    ),
                    params={"iwm_hooks": hooks_applied},
                )
            return CoupledAction(
                a_phy="dream",
                a_ment=IntentRecord(
                    reason=f"η={state.eta:.2f} 超阈值，需刷新自我模型",
                    expected_effect="η 经 LoopTrial 证据更新",
                    risk_assessment="low",
                ),
                params={"iwm_hooks": hooks_applied},
            )

        # Step 3: Dimension lifecycle scan
        dim_status = self.dims.scan(state)
        target_dims = self.dims.select_priority(dim_status)

        # Step 4: Generate candidates and select
        candidates = self._generate_candidates(state, target_dims, advice=advice)
        mem_reason = advice.get("prefer_learn_reason") or ""
        if advice.get("prefer_learn"):
            tag = f"[IWM: {mem_reason or 'prefer_learn'}]"
            for c in candidates:
                if c["action"] == "learn":
                    c["reason"] = c.get("reason", "") + f" {tag}"
                if c["action"] == "remember" and advice.get("prefer_remember_ingest"):
                    c["reason"] = c.get("reason", "") + " [IWM: memory 不可靠，优先重新摄入]"

        # Memory-trust gate: trusted memory → prefer learn; untrusted → block evolve / allow remember
        if advice.get("trust_memory_for_learn"):
            for c in candidates:
                if c["action"] == "learn":
                    c["priority_boost"] = "memory_trust"
        if advice.get("downweight_memory_ops"):
            candidates = [
                c for c in candidates
                if c["action"] != "evolve"
            ]
            if not any(c["action"] == "remember" for c in candidates):
                candidates.append({
                    "action": "remember",
                    "reason": "IWM memory 不可靠：重新摄入而非依赖蒸馏经验",
                    "expected": "恢复 PROXY 证据",
                    "risk": "low",
                    "params": {"iwm_memory": "reingest"},
                })
            if not any(c["action"] == "learn" for c in candidates):
                candidates.append({
                    "action": "learn",
                    "reason": "IWM memory 不可靠：用新轨迹重蒸馏",
                    "expected": "重建可信经验",
                    "risk": "low",
                })

        # Step 5: Select best — pre-enactment downweighted when dynamics untrusted
        depth = self.siwm.eta.adaptive_depth()
        if advice.get("degrade_to_baseline") or advice.get("downweight_pre_enactment"):
            depth = 0
        if depth > 0 and len(candidates) > 1:
            best = self._pre_enact_select(candidates, state, depth)
        else:
            best = self._heuristic_select(candidates, advice=advice)

        # Provenance (Q5)
        if self.iwm is not None:
            try:
                rec = self.iwm.record_decision(
                    action=best["action"],
                    reason=best.get("reason", ""),
                    decision_source="governor",
                    candidates=[c["action"] for c in candidates],
                    state_digest={
                        "eta": state.eta,
                        "generation": state.phi.generation,
                        "trace_count": state.phi.storage_stats.get("trace_count", 0),
                    },
                    iwm_snapshot=advice,
                    hooks_applied=hooks_applied,
                )
                self._last_provenance_id = rec.decision_id
            except Exception:
                self._last_provenance_id = ""

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
            params={**(best.get("params") or {}), "iwm_hooks": hooks_applied},
        )

    def _generate_candidates(
        self,
        state: WorldState,
        target_dims: list[EmpowermentDimension],
        advice: Optional[dict] = None,
    ) -> list[dict]:
        """Generate candidate actions based on state and target dimensions."""
        candidates = []
        advice = advice or {}
        forbid_dream = bool(advice.get("forbid_default_dream"))

        # Dream candidate only when η rising AND dream organ allowed
        if state.eta >= 0.20 and not forbid_dream:
            candidates.append({
                "action": "dream",
                "reason": f"η={state.eta:.2f}，预防性梦境刷新",
                "expected": "η 证据更新",
                "risk": "low",
            })
        elif state.eta >= 0.20 and forbid_dream:
            candidates.append({
                "action": "learn",
                "reason": f"η={state.eta:.2f} 但 IWM 禁止默认 dream，改 learn",
                "expected": "蒸馏经验替代无证据 dream",
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

        # Evolve: only when memory ops trusted AND behavior not forcing learn
        if exp_count > 10 and not advice.get("prefer_learn") and not advice.get("downweight_memory_ops"):
            candidates.append({
                "action": "evolve",
                "reason": f"经验记忆 {exp_count} 条，可尝试机制突变",
                "expected": "机制效果提升",
                "risk": "medium",
            })

        # Trusted memory organ → learn is a first-class candidate when proxy backlog exists
        if advice.get("trust_memory_for_learn") and trace_count > exp_count:
            if not any(c["action"] == "learn" for c in candidates):
                candidates.append({
                    "action": "learn",
                    "reason": f"IWM memory trusted (trust={advice.get('memory_trust')}), PROXY 未蒸馏 {trace_count - exp_count}",
                    "expected": "可信经验蒸馏",
                    "risk": "low",
                    "params": {"iwm_memory": "trusted_learn"},
                })

        # Consider maintain if memory is growing
        if trace_count > 100:
            candidates.append({
                "action": "maintain",
                "reason": f"轨迹积累 {trace_count} 条，需要清理",
                "expected": "存储优化",
                "risk": "low",
            })

        # Frontier explore bias: add underexplored categories as learn/explore nudge
        explore_bias = advice.get("explore_bias") or []
        if explore_bias and not forbid_dream:
            candidates.append({
                "action": "learn",
                "reason": f"IWM frontier 探索偏向: {explore_bias[:3]}",
                "expected": "补齐知识边界",
                "risk": "low",
                "params": {"frontier_explore": explore_bias[:3]},
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
        """Select best candidate via simplified pre-enactment."""
        best = candidates[0]
        best_score = -1.0

        for c in candidates:
            score = self._score_candidate(c, state)
            if score > best_score:
                best_score = score
                best = c

        return best

    def _heuristic_select(self, candidates: list[dict], advice: Optional[dict] = None) -> dict:
        """Heuristic selection when pre-enactment is unavailable/untrusted."""
        advice = advice or {}
        priority = {"dream": 3, "maintain": 2, "learn": 2, "evolve": 1, "remember": 1}
        if advice.get("forbid_default_dream"):
            priority["dream"] = 0
            priority["learn"] = 4
        if advice.get("prefer_learn"):
            priority["learn"] = max(priority.get("learn", 2), 3)
            priority["evolve"] = 0
        if advice.get("trust_memory_for_learn"):
            priority["learn"] = max(priority.get("learn", 2), 4)
        if advice.get("downweight_memory_ops"):
            priority["evolve"] = 0
            priority["remember"] = max(priority.get("remember", 1), 3)
            priority["learn"] = max(priority.get("learn", 2), 3)
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
        out = {
            "decision_count": self._decision_count,
            "policy_keys": list(self.policy.keys()),
            "eta": self.siwm.eta.value,
            "last_provenance_id": self._last_provenance_id,
        }
        if self.iwm is not None:
            out["iwm_hooks"] = self._last_iwm_advice.get("degrade_to_baseline")
            out["forbid_default_dream"] = self._last_iwm_advice.get("forbid_default_dream")
            out["memory_trust"] = self._last_iwm_advice.get("memory_trust")
            out["trust_memory_for_learn"] = self._last_iwm_advice.get("trust_memory_for_learn")
            out["downweight_memory_ops"] = self._last_iwm_advice.get("downweight_memory_ops")
        return out
