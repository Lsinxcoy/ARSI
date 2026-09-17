"""Empowerment Engine — makes host agents stronger.

observe → diagnose → empower → verify → learn

Based on: ARSI Whitepaper v0.8 §7.6
Inherited from SYNTHEX's weapon rack (Brain-Hands-Session).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Protocol

from arsi.foundation.schema import (
    EmpowermentDimension,
    EmpowermentOp,
    VerificationStatus,
    WorldState,
)
from arsi.mnemosyne.core import Mnemosyne
from arsi.world_model.siwm import SIWM

logger = logging.getLogger(__name__)


class AgentAdapter(Protocol):
    """Interface for interacting with a host agent."""

    def snapshot(self, agent_id: str) -> dict: ...
    def apply(self, agent_id: str, patch: dict) -> dict: ...
    def restore(self, agent_id: str, snapshot: dict) -> None: ...


class NullAdapter:
    """Fallback adapter for testing."""

    def snapshot(self, agent_id: str) -> dict:
        return {"agent_id": agent_id, "timestamp": datetime.now().isoformat()}

    def apply(self, agent_id: str, patch: dict) -> dict:
        return {"status": "applied", "patch_keys": list(patch.keys())}

    def restore(self, agent_id: str, snapshot: dict) -> None:
        pass


class EmpowermentVerifier:
    """Four-level honest verification.

    Level 1: agent behavior change observable
    Level 2: performance metrics measurable
    Level 3: sealed eval A/B verifiable
    Level 4: causal chain traceable

    NEVER blind SUCCESS — no evidence → UNKNOWN.
    """

    def verify(
        self,
        agent_id: str,
        weapon: dict,
        pre_snapshot: dict,
        post_snapshot: dict,
    ) -> tuple[VerificationStatus, dict]:
        """Verify empowerment effect. Returns (status, evidence)."""
        evidence: dict = {}

        # Level 1: Behavior change
        behavior_changed = pre_snapshot != post_snapshot
        evidence["behavior_changed"] = behavior_changed

        # Level 2: Performance (would need metrics — simplified)
        evidence["performance_measured"] = False

        # Level 3: Sealed eval (would need sealed evaluator — simplified)
        evidence["sealed_eval_done"] = False

        # Level 4: Causal chain
        evidence["causal_chain_available"] = bool(weapon.get("causal_chain"))

        # Determine status — NEVER blind SUCCESS
        if not behavior_changed and not evidence["sealed_eval_done"]:
            return VerificationStatus.UNKNOWN, {
                **evidence,
                "reason": "无行为变化证据，拒绝盲标 SUCCESS",
            }

        if behavior_changed and evidence["causal_chain_available"]:
            return VerificationStatus.SUCCESS, evidence

        if behavior_changed:
            return VerificationStatus.PARTIAL, evidence

        return VerificationStatus.UNKNOWN, {
            **evidence,
            "reason": "无可验证证据",
        }


class EmpowermentEngine:
    """Empowerment execution loop.

    Based on SYNTHEX's execute → record → adapt, upgraded to:
    observe → diagnose → empower → verify → learn
    """

    def __init__(
        self,
        mnemosyne: Mnemosyne,
        siwm: SIWM,
        adapter: Optional[AgentAdapter] = None,
    ):
        self.mnemosyne = mnemosyne
        self.siwm = siwm
        self.adapter = adapter or NullAdapter()
        self.verifier = EmpowermentVerifier()
        self._empowerment_count = 0

    def empower(self, agent_id: str, state: WorldState) -> EmpowermentOp:
        """Execute one empowerment cycle."""
        self._empowerment_count += 1

        # 1. Observe: read agent's behavior traces
        traces = self.mnemosyne.store.get_recent_traces(n=50, agent_id=agent_id)

        # 2. Diagnose: identify weakness
        weakness = self._diagnose(traces, state)

        # 3. Select empowerment weapon
        weapon = self._select_weapon(weakness, agent_id)

        # 4. Pre-snapshot
        pre_snapshot = self.adapter.snapshot(agent_id)

        # 5. Apply empowerment
        try:
            self.adapter.apply(agent_id, weapon["patch"])
        except Exception as e:
            logger.error(f"Empowerment apply failed: {e}")
            return EmpowermentOp(
                dimension=weapon["dimension"],
                target_agent=agent_id,
                content=weapon["patch"],
                verification=VerificationStatus.FAILURE,
                verification_evidence={"error": str(e)},
            )

        # 6. Post-snapshot
        post_snapshot = self.adapter.snapshot(agent_id)

        # 7. Verify (NEVER blind SUCCESS)
        status, evidence = self.verifier.verify(
            agent_id=agent_id,
            weapon=weapon,
            pre_snapshot=pre_snapshot,
            post_snapshot=post_snapshot,
        )

        # 8. Record
        op = EmpowermentOp(
            dimension=weapon["dimension"],
            target_agent=agent_id,
            content=weapon["patch"],
            pre_snapshot=pre_snapshot,
            post_snapshot=post_snapshot,
            verification=status,
            verification_evidence=evidence,
        )

        self.mnemosyne.record_empowerment(op, causal_chain_id=weapon.get("causal_chain", ""))
        return op

    def rollback(self, op: EmpowermentOp) -> bool:
        """Rollback an empowerment operation (atomic)."""
        if op.rolled_back:
            return False
        try:
            self.adapter.restore(op.target_agent, op.pre_snapshot)
            op.rolled_back = True
            logger.info(f"Rolled back empowerment {op.id}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def _diagnose(self, traces: list[dict], state: WorldState) -> dict:
        """Diagnose agent's weakness from traces."""
        if not traces:
            return {"dimension": EmpowermentDimension.KNOWLEDGE, "reason": "无轨迹数据"}

        # Analyze action patterns
        actions = [t.get("action", "") for t in traces]
        outcomes = [t.get("outcome", "") for t in traces]

        success_rate = sum(1 for o in outcomes if "success" in o.lower()) / max(len(outcomes), 1)

        if success_rate < 0.5:
            return {
                "dimension": EmpowermentDimension.HARNESS,
                "reason": f"成功率过低 ({success_rate:.0%})，需要脚手架改进",
            }

        # Check for repeated failures in specific action
        action_outcomes: dict[str, list[str]] = {}
        for t in traces:
            a = t.get("action", "")
            o = t.get("outcome", "")
            action_outcomes.setdefault(a, []).append(o)

        for action, outs in action_outcomes.items():
            fail_rate = sum(1 for o in outs if "fail" in o.lower()) / max(len(outs), 1)
            if fail_rate > 0.5:
                return {
                    "dimension": EmpowermentDimension.DECOMPOSITION,
                    "reason": f"动作 '{action}' 失败率 {fail_rate:.0%}，需要分解策略改进",
                }

        return {
            "dimension": EmpowermentDimension.KNOWLEDGE,
            "reason": "基础成功率正常，注入知识",
        }

    def _select_weapon(self, weakness: dict, agent_id: str) -> dict:
        """Select empowerment weapon based on diagnosed weakness."""
        dim = weakness["dimension"]

        # Search experience memory for relevant past lessons
        experiences = self.mnemosyne.search_experience(
            query=dim.value + " " + weakness.get("reason", ""),
            agent_id=agent_id,
        )

        patch = {
            "type": dim.value,
            "recommendation": weakness["reason"],
            "experience_refs": [e.id for e in experiences[:3]],
        }

        return {
            "dimension": dim,
            "patch": patch,
            "causal_chain": experiences[0].id if experiences else "",
        }

    @property
    def stats(self) -> dict:
        return {
            "empowerment_count": self._empowerment_count,
        }
