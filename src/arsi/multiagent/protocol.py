"""Multi-agent active protocol — ARSI orchestrates host agents (M1–M4).

Contract:
  1. register(agent_id, role, capabilities)
  2. ARSI issues ASSIGN + structured BRIEF (via ARSIInterface.brief)
  3. Host executes; posts RESULT
  4. ARSI.report() ingests outcome + external effect anchor
  5. Iron laws: freeze blocks dispatch; unmeasured organs do not drive control

Not a hive copy of SYNTHEX — thin orchestration over existing brief/report.
"""
from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Deque, Optional

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    WORKER = "worker"
    TOOL_RUNNER = "tool_runner"
    RESEARCHER = "researcher"
    UNKNOWN = "unknown"


class AgentStatus(str, Enum):
    REGISTERED = "registered"
    ASSIGNED = "assigned"
    BUSY = "busy"
    IDLE = "idle"
    UNMEASURED = "unmeasured"
    FAILED = "failed"
    FROZEN = "frozen"


class MsgType(str, Enum):
    REGISTER = "register"
    ASSIGN = "assign"
    BRIEF = "brief"
    RESULT = "result"
    HEARTBEAT = "heartbeat"
    FREEZE = "freeze"


@dataclass
class ProtocolMessage:
    msg_id: str = ""
    msg_type: str = MsgType.ASSIGN.value
    agent_id: str = ""
    task_id: str = ""
    payload: dict = field(default_factory=dict)
    timestamp: str = ""
    source: str = "arsi"

    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentRecord:
    agent_id: str
    role: str = AgentRole.WORKER.value
    capabilities: list[str] = field(default_factory=list)
    status: str = AgentStatus.UNMEASURED.value
    tasks_assigned: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_outcome: str = ""
    last_brief_id: str = ""
    organ_status: str = "unmeasured"  # M3
    registered_at: str = ""

    def __post_init__(self):
        if not self.registered_at:
            self.registered_at = datetime.now().isoformat()

    @property
    def success_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        if total <= 0:
            return 0.0
        return self.tasks_completed / total

    def to_dict(self) -> dict:
        d = asdict(self)
        d["success_rate"] = round(self.success_rate, 4)
        return d


class MultiAgentOrchestrator:
    """Active multi-agent loop on top of ARSIBrief / ARSIReport."""

    def __init__(self, arsi=None, interface=None, max_queue: int = 200):
        self.arsi = arsi
        self.interface = interface
        self.agents: dict[str, AgentRecord] = {}
        self.outbox: Deque[ProtocolMessage] = deque(maxlen=max_queue)
        self.inbox: Deque[ProtocolMessage] = deque(maxlen=max_queue)
        self.history: Deque[dict] = deque(maxlen=500)
        self._frozen = False
        self._dispatch_count = 0
        self._result_count = 0

    def register(
        self,
        agent_id: str,
        role: str = AgentRole.WORKER.value,
        capabilities: Optional[list[str]] = None,
    ) -> AgentRecord:
        rec = AgentRecord(
            agent_id=agent_id,
            role=role,
            capabilities=list(capabilities or []),
            status=AgentStatus.UNMEASURED.value,
            organ_status="unmeasured",
        )
        self.agents[agent_id] = rec
        msg = ProtocolMessage(
            msg_type=MsgType.REGISTER.value,
            agent_id=agent_id,
            payload={"role": role, "capabilities": rec.capabilities},
        )
        self.outbox.append(msg)
        self.history.append({"event": "register", "agent_id": agent_id, "ts": msg.timestamp})
        logger.info(f"MultiAgent registered {agent_id} role={role}")
        return rec

    def _iron_laws_block(self) -> Optional[str]:
        if self._frozen:
            return "orchestrator_frozen"
        if self.arsi is not None:
            try:
                state = self.arsi.siwm.refresh_state()
                if self.arsi.iron_laws.violated(state):
                    self._frozen = True
                    return "iron_law_violated_freeze"
            except Exception as e:
                return f"iron_law_check_error:{e}"
        return None

    def dispatch(self, agent_id: str, task_description: str, payload: Optional[dict] = None) -> dict:
        """Issue ASSIGN + structured BRIEF for one agent task."""
        block = self._iron_laws_block()
        if block:
            return {"dispatched": False, "reason": block}

        rec = self.agents.get(agent_id)
        if rec is None:
            return {"dispatched": False, "reason": "agent_not_registered"}
        if rec.status == AgentStatus.FROZEN.value:
            return {"dispatched": False, "reason": "agent_frozen"}

        task_id = f"task_{uuid.uuid4().hex[:10]}"
        brief_id = ""
        brief_text = ""
        if self.interface is not None:
            try:
                brief = self.interface.brief(task_description, agent_id)
                brief_id = brief.brief_id
                brief_text = brief.format_for_agent()
            except Exception as e:
                logger.warning(f"brief generation failed: {e}")
                brief_text = f"# ARSI assign\nTask: {task_description}"

        assign = ProtocolMessage(
            msg_type=MsgType.ASSIGN.value,
            agent_id=agent_id,
            task_id=task_id,
            payload={
                "task_description": task_description,
                "brief_id": brief_id,
                "capabilities_required": (payload or {}).get("capabilities_required", []),
                **(payload or {}),
            },
        )
        brief_msg = ProtocolMessage(
            msg_type=MsgType.BRIEF.value,
            agent_id=agent_id,
            task_id=task_id,
            payload={"brief_id": brief_id, "text": brief_text, "policy": "structured"},
        )
        self.outbox.append(assign)
        self.outbox.append(brief_msg)

        rec.status = AgentStatus.ASSIGNED.value
        rec.tasks_assigned += 1
        rec.last_brief_id = brief_id
        rec.organ_status = "unmeasured"  # M3 until result evidence
        self._dispatch_count += 1
        self.history.append({
            "event": "dispatch",
            "agent_id": agent_id,
            "task_id": task_id,
            "brief_id": brief_id,
            "ts": assign.timestamp,
        })
        return {
            "dispatched": True,
            "task_id": task_id,
            "brief_id": brief_id,
            "assign": assign.to_dict(),
        }

    def submit_result(
        self,
        agent_id: str,
        task_id: str,
        task_description: str,
        outcome: str = "failure",
        effect: float = 0.0,
        skills_used: Optional[list[str]] = None,
        notes: str = "",
        recommendations_followed: Optional[list[str]] = None,
        recommendations_ignored: Optional[list[str]] = None,
    ) -> dict:
        """Ingest host RESULT → ARSI report + external effect + organ evidence."""
        rec = self.agents.get(agent_id)
        if rec is None:
            return {"accepted": False, "reason": "agent_not_registered"}

        success = "success" in (outcome or "").lower()
        rec.tasks_completed += 1 if success else 0
        rec.tasks_failed += 0 if success else 1
        rec.last_outcome = outcome
        rec.status = AgentStatus.IDLE.value if success else AgentStatus.FAILED.value
        # M3: after ≥3 outcomes, organ becomes measurable
        measured = (rec.tasks_completed + rec.tasks_failed) >= 3
        rec.organ_status = "measured" if measured else "unmeasured"

        ingest = {"status": "no_interface"}
        if self.interface is not None:
            try:
                from arsi.adapters.bidirectional_interface import ARSIReport
                ingest = self.interface.report(ARSIReport(
                    brief_id=rec.last_brief_id or "",
                    agent_id=agent_id,
                    task_description=task_description,
                    outcome=outcome,
                    effect=float(effect),
                    skills_used=list(skills_used or []),
                    recommendations_followed=list(recommendations_followed or []),
                    recommendations_ignored=list(recommendations_ignored or []),
                    notes=notes,
                ))
            except Exception as e:
                ingest = {"status": "report_error", "error": str(e)}

        # S5 external effect anchor from host outcome
        anchor = None
        if self.arsi is not None and getattr(self.arsi, "store", None) is not None:
            try:
                from arsi.foundation.effect_anchor import record_external_effect
                anchor = record_external_effect(
                    self.arsi.store,
                    mechanism=f"multiagent:{agent_id}",
                    effect=float(effect),
                    source="host_outcome",
                    evidence_ref=f"{agent_id}:{task_id}",
                ).to_dict()
            except Exception as e:
                anchor = {"error": str(e)}

        # IWM: bind host success as portfolio/memory-adjacent evidence when measured
        iwm_bind = None
        if self.arsi is not None and getattr(self.arsi, "iwm", None) is not None and measured:
            try:
                self.arsi.iwm.observe_memory(success, note=f"multiagent:{agent_id}:{outcome}")
                iwm_bind = "observe_memory"
            except Exception:
                iwm_bind = None

        self._result_count += 1
        msg = ProtocolMessage(
            msg_type=MsgType.RESULT.value,
            agent_id=agent_id,
            task_id=task_id,
            payload={"outcome": outcome, "effect": effect, "organ_status": rec.organ_status},
        )
        self.inbox.append(msg)
        self.history.append({
            "event": "result",
            "agent_id": agent_id,
            "task_id": task_id,
            "outcome": outcome,
            "effect": effect,
            "organ_status": rec.organ_status,
            "ts": msg.timestamp,
        })
        return {
            "accepted": True,
            "agent": rec.to_dict(),
            "arsi_ingest": ingest,
            "effect_anchor": anchor,
            "iwm_bind": iwm_bind,
        }

    def freeze(self, reason: str = "") -> None:
        self._frozen = True
        self.outbox.append(ProtocolMessage(
            msg_type=MsgType.FREEZE.value,
            agent_id="*",
            payload={"reason": reason or "manual"},
        ))

    def cycle(self) -> dict:
        """One orchestration pass: idle agents get assignments from ARSI frontier."""
        block = self._iron_laws_block()
        if block:
            return {"status": "blocked", "reason": block}

        idle = [a for a in self.agents.values() if a.status in (
            AgentStatus.IDLE.value,
            AgentStatus.REGISTERED.value,
            AgentStatus.UNMEASURED.value,
        )]
        dispatched = []
        # Prefer measurable agents; unmeasured get exploratory tasks
        idle.sort(key=lambda a: (0 if a.organ_status == "measured" else 1, -a.success_rate))
        for rec in idle[:2]:
            task = self._suggest_task(rec)
            res = self.dispatch(rec.agent_id, task)
            if res.get("dispatched"):
                dispatched.append(res["task_id"])
        return {
            "status": "ok",
            "dispatched": dispatched,
            "agents": {k: v.to_dict() for k, v in self.agents.items()},
            "frozen": self._frozen,
        }

    def _suggest_task(self, rec: AgentRecord) -> str:
        if rec.organ_status != "measured":
            return f"[calibration] structured self-report for {rec.agent_id} capabilities={rec.capabilities[:3]}"
        if self.arsi is not None and getattr(self.arsi, "iwm", None) is not None:
            try:
                bias = self.arsi.iwm.frontier.explore_bias()[:2]
                if bias:
                    return f"[explore] address frontier weak dims: {bias}"
            except Exception:
                pass
        return f"[worker] generic improvement task for {rec.agent_id}"

    def health(self) -> dict:
        unmeasured = [a.agent_id for a in self.agents.values() if a.organ_status == "unmeasured"]
        return {
            "multi_agent": {
                "frozen": self._frozen,
                "agent_count": len(self.agents),
                "dispatch_count": self._dispatch_count,
                "result_count": self._result_count,
                "outbox": len(self.outbox),
                "inbox": len(self.inbox),
                "unmeasured_agents": unmeasured,
                "agents": {k: v.to_dict() for k, v in self.agents.items()},
            }
        }

    def stats(self) -> dict:
        return self.health()["multi_agent"]
