"""ARSI Bidirectional Interface Protocol — agents pass through ARSI on every task.

Protocol (empowerment-max · dual boundary):
  1. Agent calls brief() before task → Empowerment Pack (skills/tools/manifests/
     fail-recovery/drawbacks/what-if) + structured strategy. Host face MAY be
     semantic and actionable.
  2. Agent executes task
  3. Agent calls report() after task → actionable schema + negative-epistemology
     claim gate (never blind SUCCESS)

Dual boundaries:
  - ARSI internal replay_selection still forbids semantic guidance (P-R9)
  - claims only no_known_defect / verified_contract / measured / unverified
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from arsi.core import ARSI
from arsi.foundation.schema import MemoryRecord, MemoryZone

logger = logging.getLogger(__name__)

# Feedback file for MiMo (can be extended to other agents)
FEEDBACK_DIR = Path.home() / ".local" / "share" / "mimocode" / "memory" / "arsi_feedback"


class ARSIBrief:
    """What ARSI tells an agent before a task.

    Host face is empowerment-max (skills, tools, manifests, fail recovery,
    drawbacks, what-if). Internal ARSI selection remains free of this prose.
    brief_policy:
      - structured (default): empowerment pack + structured stats; free-form
        past lessons stay meta_only.
      - full: also dump recommendations / past lessons (human CLI / debug).
    """

    POLICY_STRUCTURED = "structured"
    POLICY_FULL = "full"

    def __init__(
        self,
        task_description: str,
        agent_id: str,
        recommendations: list[str],
        relevant_skills: list[str],
        warnings: list[str],
        past_lessons: list[str],
        confidence: float = 0.5,
        history_simulator: Optional[dict] = None,
        policy: str = "structured",
        empower_pack: Optional[dict] = None,
        working_state: Optional[dict] = None,
    ):
        self.task_description = task_description
        self.agent_id = agent_id
        self.recommendations = recommendations
        self.relevant_skills = relevant_skills
        self.warnings = warnings
        self.past_lessons = past_lessons
        self.confidence = confidence
        self.history_simulator = history_simulator  # Dream-RSI: queryable history
        self.policy = policy or self.POLICY_STRUCTURED
        self.empower_pack = empower_pack  # max host capability surface
        self.working_state = working_state  # Recuris WM snapshot
        self.timestamp = datetime.now().isoformat()
        self.brief_id = f"brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def query_history(self, task_type: str) -> dict:
        """Dream-RSI: Agent can query historical performance.

        Returns structured data, not advice text. No directional fields.
        """
        if not self.history_simulator or not self.history_simulator.get("available"):
            return {"available": False, "reason": "no_history_data"}

        hs = self.history_simulator
        return {
            "available": True,
            "task_type": task_type,
            "historical_success_rate": hs.get("success_rate", 0),
            "sample_size": hs.get("sample_size", 0),
            "common_failure_patterns": hs.get("common_failure_patterns", []),
            "score_stats": hs.get("score_stats", {}),
            "confidence": hs.get("confidence", 0),
        }

    def to_dict(self) -> dict:
        return {
            "brief_id": self.brief_id,
            "agent_id": self.agent_id,
            "task": self.task_description,
            "policy": self.policy,
            "recommendations": self.recommendations,
            "relevant_skills": self.relevant_skills,
            "warnings": self.warnings,
            "past_lessons": self.past_lessons,
            "confidence": self.confidence,
            "history_available": self.history_simulator is not None and self.history_simulator.get("available", False),
            "history_simulator": self.history_simulator,
            "meta_only": {
                "recommendations": self.recommendations,
                "past_lessons": self.past_lessons,
            },
            "empower_pack": self.empower_pack,
            "working_state": self.working_state,
            "timestamp": self.timestamp,
        }

    def format_for_agent(self, full: bool = False) -> str:
        """Format brief for agent consumption.

        B-line (SoL-Pi channel):
          - structured policy: no Recommendations / Past Lessons (§5.1)
          - large payload → ObservationPack handle + excerpt
          - evidence receipt block when channel_receipt is attached
        """
        lines = [
            f"# ARSI Brief [{self.brief_id}]",
            f"Agent: {self.agent_id}  Task: {self.task_description}",
            f"Confidence: {self.confidence:.2f}",
            f"Policy: {self.policy}",
            "",
        ]
        # Pin capture contract at top when terminal failures are hot
        ep = self.empower_pack
        top = []
        try:
            top = [c.get("label") for c in ((getattr(ep, "preflight", None) or {}).get("top_clusters") or [])]
        except Exception:
            top = []
        if any("terminal" in str(x) for x in top) or True:
            lines.append("## HARD CONTRACT (must before any SUCCESS)")
            lines.append("  - terminal/execute_code fail → measurements MUST include exit_code + stderr_tail")
            lines.append("  - fail_class=tool_error_terminal|execute; one recovery; never verbal done")
            lines.append("")
        if self.relevant_skills:
            lines.append("## Relevant Skills")
            for s in self.relevant_skills:
                lines.append(f"  - {s}")
            lines.append("")
        if self.warnings:
            lines.append("## Warnings")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
            lines.append("")

        # Empowerment-max pack (host face semantic OK)
        if self.empower_pack:
            pack = self.empower_pack
            if hasattr(pack, "as_structured_block"):
                lines.extend(pack.as_structured_block())
            elif isinstance(pack, dict):
                from arsi.adapters.host_empower import EmpowermentPack

                lines.extend(EmpowermentPack(**{k: pack.get(k) for k in EmpowermentPack.__dataclass_fields__ if k in pack}).as_structured_block())
            lines.append("")

        # P-c bounded context for long lesson/recommendation dumps
        try:
            from arsi.harness.runtime_wiring import brief_compress
            if self.past_lessons or len(self.recommendations or []) > 8:
                folded = brief_compress(list(self.past_lessons or []) + list(self.recommendations or []))
                lines.append("## Compressed prior (bounded)")
                lines.append(folded[:800])
        except Exception:
            pass

        # Fail-handling (CM-fc23371019 + P1-2 gate_reject + tool_error) - structured, executable
        lines.append("## Fail-handling")
        lines.append("  - On tool failure (esp. terminal/execute_code): fail_class=tool_error + tool_id.")
        lines.append("  - One recovery only: backoff retry OR switch_alternate_tool (execute_code↔terminal).")
        lines.append("  - Do not repeat the identical failing call without a changed precondition.")
        lines.append("  - On gate/synthex reject: fail_class=gate_reject; log gate_id+candidate_id; ONE repair pass only.")
        lines.append("  - Compile (synthex/mech): tag candidate_id; ONE recompile; 2nd fail same id → abandon_and_log.")
        lines.append("  - Report fail_class + recovery_attempted + recovery_worked; never mark tool_loop/gate_reject as success.")
        lines.append("")

        hs = self.history_simulator or {}
        if hs.get("available"):
            lines.append("## History Stats (structured, not guidance)")
            lines.append(f"  - success_rate: {hs.get('success_rate', 0)}")
            lines.append(f"  - sample_size: {hs.get('sample_size', 0)}")
            stats = hs.get("score_stats") or {}
            if stats:
                lines.append(f"  - score: {stats}")
            patterns = hs.get("common_failure_patterns") or []
            if patterns:
                lines.append(f"  - failure_pattern_count: {len(patterns)}")
            lines.append("")

        if self.policy == self.POLICY_FULL:
            if self.recommendations:
                lines.append("## Recommendations")
                for r in self.recommendations:
                    lines.append(f"  - {r}")
                lines.append("")
            if self.past_lessons:
                lines.append("## Past Lessons")
                for l in self.past_lessons:
                    lines.append(f"  - {l}")
        else:
            lines.append("## Note")
            lines.append("  History is provided as structured stats only (no direction advice).")

        # A-line: IWM strategy structured facts
        strategy = getattr(self, "channel_strategy", None)
        if strategy is not None:
            if hasattr(strategy, "as_structured_block"):
                lines.extend(strategy.as_structured_block())
                lines.append("")
            elif isinstance(strategy, dict):
                lines.append("## IWM Strategy (structured facts)")
                for k in ("focus", "confidence", "control_flags", "operational_warnings"):
                    if k in strategy:
                        lines.append(f"- {k}: {strategy[k]}")
                lines.append("")

        # Channel evidence (B-line)
        receipt = getattr(self, "channel_receipt", None)
        if receipt:
            lines.append("## Evidence Receipt (channel)")
            if hasattr(receipt, "brief_snippet"):
                lines.append(receipt.brief_snippet())
            elif isinstance(receipt, dict):
                lines.append(f"- id: {receipt.get('receipt_id')}")
                lines.append(f"- verified: {receipt.get('verified')} ({receipt.get('reason')})")
                for q in (receipt.get("quotes") or [])[:3]:
                    lines.append(f"  - {q}")
            lines.append("")

        body = "\n".join(lines)
        self._last_full_body = body
        if full:
            return body
        # ObservationPack: large brief → handle for host, full text archived
        try:
            from arsi.foundation.observation_pack import pack_for_host
            packed = pack_for_host(body, kind=f"brief_{self.agent_id}")
            if packed.startswith("# ARSI Observation Handle"):
                return packed + "\n\n## Full brief handle note\nOriginal archived; retrieve via handle.\n"
            return body
        except Exception:
            return body


class ARSIReport:
    """What an agent tells ARSI after a task (actionable schema + claim gate).

    Actionable fields feed AEGIS digester labels and recovery learning.
    `claim` (optional) is screened by negative epistemology — never blind SUCCESS.
    """

    def __init__(
        self,
        brief_id: str,
        agent_id: str,
        task_description: str,
        outcome: str,
        effect: float,
        skills_used: list[str],
        recommendations_followed: list[str],
        recommendations_ignored: list[str],
        notes: str = "",
        fail_class: str = "",
        recovery_attempted: str = "",
        recovery_worked: Optional[bool] = None,
        acceptance_evidence: str = "",
        manifest_ids_used: list[str] | None = None,
        compile_checked: Optional[bool] = None,
        claim: Optional[dict] = None,
        measurements: Optional[dict] = None,
        wm_updates: Optional[list] = None,
    ):
        self.brief_id = brief_id
        self.agent_id = agent_id
        self.task_description = task_description
        self.outcome = outcome
        self.effect = effect
        self.skills_used = skills_used
        self.recommendations_followed = recommendations_followed
        self.recommendations_ignored = recommendations_ignored
        self.notes = notes
        self.fail_class = fail_class or ""
        self.recovery_attempted = recovery_attempted or ""
        self.recovery_worked = recovery_worked
        self.acceptance_evidence = acceptance_evidence or ""
        self.manifest_ids_used = list(manifest_ids_used or [])
        self.compile_checked = compile_checked
        self.claim = claim  # {kind, subject, drawbacks_checked, contract_id, evidence, raw}
        self.measurements = dict(measurements or {})
        self.wm_updates = list(wm_updates or [])  # proposed StateUpdate dicts
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "brief_id": self.brief_id,
            "agent_id": self.agent_id,
            "task_description": self.task_description,
            "outcome": self.outcome,
            "effect": self.effect,
            "skills_used": self.skills_used,
            "fail_class": self.fail_class,
            "recovery_attempted": self.recovery_attempted,
            "recovery_worked": self.recovery_worked,
            "acceptance_evidence": self.acceptance_evidence,
            "manifest_ids_used": self.manifest_ids_used,
            "compile_checked": self.compile_checked,
            "claim": self.claim,
            "measurements": self.measurements,
            "wm_updates": self.wm_updates,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }


class ARSIInterface:
    """Bidirectional interface between ARSI and agents.

    Usage:
        interface = ARSIInterface(arsi)

        # Agent asks before task
        brief = interface.brief("Fix the bug in parser.py", "hermes")
        print(brief.format_for_agent())

        # Agent reports after task
        interface.report(ARSIReport(
            brief_id=brief.brief_id,
            agent_id="hermes",
            task_description="Fix the bug in parser.py",
            outcome="success",
            effect=0.8,
            skills_used=["arsi-metacognition"],
            recommendations_followed=["先跑测试再改代码"],
            recommendations_ignored=["连续失败后切换策略"],
        ))
    """

    def __init__(self, arsi: ARSI, brief_policy: str = ARSIBrief.POLICY_STRUCTURED):
        self.arsi = arsi
        self.brief_policy = brief_policy
        self._brief_count = 0
        self._report_count = 0
        self._active_briefs: dict[str, ARSIBrief] = {}

    def brief(
        self,
        task_description: str,
        agent_id: str,
        pull: Optional[list] = None,
    ) -> ARSIBrief:
        """Generate a brief for an agent about to execute a task.

        pull: optional read-only capability pulls, e.g. ["harness_preflight"].
              Never triggers evolve/apply.
        """
        self._brief_count += 1

        # 1. Search relevant experiences
        relevant_exp = self.arsi.mnemosyne.search_experience(
            query=task_description,
            agent_id=agent_id,
            k=5,
        )

        # 2. Search cross-agent lessons
        cross_lessons = self.arsi.mnemosyne.search_cross_agent(
            query=task_description,
            exclude_agent=agent_id,
            k=3,
        )

        # 3. Check dimension gaps for warnings
        warnings = self._generate_warnings(agent_id)

        # 4. Generate recommendations from experience
        recommendations = self._generate_recommendations(
            task_description, relevant_exp, cross_lessons
        )

        # 5. Identify relevant skills
        skills = self._identify_skills(task_description)

        # 6. Past lessons
        lessons = self._extract_lessons(cross_lessons)

        # 7. Confidence based on experience volume
        confidence = min(1.0, len(relevant_exp) / 10.0)

        # 8. Dream-RSI: Build history simulator from discovery tree
        history_simulator = self._build_history_simulator(task_description)

        # 9. A-line: IWM host empowerment strategy (structured facts)
        strategy = None
        try:
            from arsi.iwm.host_strategy import build_host_strategy
            strategy = build_host_strategy(self.arsi, agent_id=agent_id)
            # confidence blend: experience volume + IWM strategy confidence
            confidence = max(0.05, min(0.95, 0.5 * confidence + 0.5 * strategy.confidence))
            # skill priors from IWM join task skills
            for s in strategy.skill_priorities:
                if s not in skills:
                    skills.append(s)
            for w in strategy.operational_warnings:
                tag = f"[IWM:{w}]"
                if tag not in warnings:
                    warnings.append(tag)
        except Exception as e:
            logger.warning(f"IWM host strategy failed: {e}")

        brief = ARSIBrief(
            task_description=task_description,
            agent_id=agent_id,
            recommendations=recommendations,
            relevant_skills=skills,
            warnings=warnings,
            past_lessons=lessons,
            confidence=confidence,
            history_simulator=history_simulator,
            policy=self.brief_policy,
        )
        # Empowerment-max pack (skills/tools/manifests/fail-recovery/drawbacks/what-if)
        try:
            from arsi.adapters.host_empower import build_empowerment_pack, harness_preflight

            brief.empower_pack = build_empowerment_pack(self.arsi, agent_id=agent_id)
            if pull and "harness_preflight" in pull:
                brief.empower_pack.preflight = harness_preflight(self.arsi, agent_id=agent_id)
            for s in brief.empower_pack.skill_kit:
                name = s.get("skill") if isinstance(s, dict) else str(s)
                if name and name not in brief.relevant_skills:
                    brief.relevant_skills.append(name)
        except Exception as e:
            logger.warning(f"empowerment pack failed: {e}")
            brief.empower_pack = None

        # Recuris R0: structured working state (goals pending/done/blocked)
        try:
            from arsi.iwm.working_memory import WorkingState

            goals = self._goals_from_task(task_description)
            ws = WorkingState.init_from_task(task_description, goals, task_id=brief.brief_id)
            brief.working_state = ws.snapshot()
        except Exception as e:
            logger.warning(f"working state init failed: {e}")
            brief.working_state = None
        if strategy is not None:
            brief.channel_strategy = strategy
            if self.brief_policy == ARSIBrief.POLICY_FULL:
                meta = strategy.meta_only_recommendations()
                if meta:
                    brief.recommendations = list(brief.recommendations) + meta

        self._active_briefs[brief.brief_id] = brief

        # Write to feedback file
        self._write_feedback(brief)

        # Record in Mnemosyne
        self.arsi.store.write_memory(MemoryRecord(
            zone=MemoryZone.PROXY,
            content=f"[brief] {agent_id}: {task_description[:50]}",
            tags=["brief", agent_id],
            agent_id=agent_id,
            generation=self.arsi.store.current_generation,
        ))

        logger.info(f"Brief generated for {agent_id}: {brief.brief_id}")
        return brief

    def report(self, report: ARSIReport) -> dict:
        """Process an agent's task report.

        This is what ARSI "hears" from the agent after it works.
        """
        self._report_count += 1

        # 0aa) tool_error_terminal: require exit_code/stderr capture (CM capture)
        try:
            fc = str(getattr(report, "fail_class", "") or "")
            m = report.measurements or {}
            if fc in ("tool_error_terminal", "tool_error") and "terminal" in f"{report.task_description} {report.notes}":
                if not (m.get("exit_code") is not None or m.get("stderr_tail") or m.get("stdout_tail")):
                    report.fail_class = "tool_error_terminal"
                    # do not invent success; mark capture missing
                    m = dict(m)
                    m["capture_missing"] = True
                    report.measurements = m
        except Exception:
            pass

        # 0a) compile retry ring: abandoned candidate_id is not a new attempt
        try:
            from arsi.harness.compile_gate import extract_candidate_id, is_reversion_blocked, note_trace_candidate

            blob = f"{report.task_description} {report.notes} {report.measurements}"
            cid = extract_candidate_id(blob)
            if cid:
                _, abandon, why = note_trace_candidate(blob)
                blocked, bwhy = is_reversion_blocked(blob)
                if abandon or blocked:
                    report.outcome = "unknown"
                    report.effect = min(float(report.effect or 0.0), 0.0)
                    report.fail_class = report.fail_class or "compile_retry_ring"
                    report.measurements = dict(report.measurements or {})
                    report.measurements["reversion_blocked"] = True
                    report.measurements["candidate_id"] = cid
        except Exception:
            pass

        # 0b) Recuris WM: commit proposed updates only with checker evidence
        wm_results = []
        try:
            from arsi.iwm.working_memory import StateUpdate, WorkingState

            ws = WorkingState.init_from_task(report.task_description, [], task_id=report.brief_id)
            for raw in getattr(report, "wm_updates", []) or []:
                u = StateUpdate(
                    goal_id=str((raw or {}).get("goal_id") or "g0"),
                    new_status=str((raw or {}).get("new_status") or "pending"),
                    evidence=list((raw or {}).get("evidence") or []),
                    blocker=str((raw or {}).get("blocker") or ""),
                    claimed=bool((raw or {}).get("claimed", True)),
                )
                wm_results.append(ws.propose_update(u))
        except Exception as e:
            wm_results = [{"accepted": False, "reason": f"wm_error:{e}"}]

        # 0b) Hard-abandon: re-version of exhausted candidate_id is refused
        try:
            from arsi.harness.compile_gate import is_reversion_blocked

            blocked, bwhy = is_reversion_blocked(
                f"{report.task_description} {report.notes} {report.measurements} {getattr(report, 'fail_class', '')}"
            )
            if blocked:
                report.outcome = "unknown"
                report.effect = min(float(report.effect or 0.0), 0.0)
                report.fail_class = report.fail_class or "compile_retry_ring"
        except Exception:
            bwhy = ""

        # 0) Negative epistemology claim gate (never blind SUCCESS)
        claim_gate = {"accepted": True, "reason": "no_claim"}
        if getattr(report, "claim", None):
            try:
                from arsi.harness.epistemic import Claim, admit_claim

                c = report.claim or {}
                ok, why = admit_claim(
                    Claim(
                        kind=str(c.get("kind") or "unverified"),
                        subject=str(c.get("subject") or report.agent_id),
                        drawbacks_checked=list(c.get("drawbacks_checked") or []),
                        contract_id=str(c.get("contract_id") or ""),
                        evidence=str(c.get("evidence") or ""),
                        metric=c.get("metric"),
                        raw=str(c.get("raw") or report.notes or ""),
                    )
                )
                claim_gate = {"accepted": ok, "reason": why}
                if not ok:
                    # demote outcome — never store forbidden claim as success
                    report.outcome = "unknown"
                    report.effect = min(float(report.effect or 0.0), 0.0)
            except Exception as e:
                claim_gate = {"accepted": False, "reason": f"claim_gate_error:{e}"}

        # 1) Record the trace (actionable schema for Digester)
        self.arsi.ingest_trace(
            agent_id=report.agent_id,
            action=f"task:{report.task_description[:40]}",
            outcome=report.outcome,
            effect=report.effect,
            params={
                "brief_id": report.brief_id,
                "skills_used": report.skills_used,
                "recommendations_followed": report.recommendations_followed,
                "recommendations_ignored": report.recommendations_ignored,
                "notes": report.notes,
                "source": "arsi_interface",
                "fail_class": getattr(report, "fail_class", ""),
                "recovery_attempted": getattr(report, "recovery_attempted", ""),
                "recovery_worked": getattr(report, "recovery_worked", None),
                "acceptance_evidence": getattr(report, "acceptance_evidence", ""),
                "manifest_ids_used": getattr(report, "manifest_ids_used", []),
                "compile_checked": getattr(report, "compile_checked", None),
                "measurements": getattr(report, "measurements", {}),
                "claim_gate": claim_gate,
            },
        )

        # 2. Evaluate recommendation effectiveness
        brief = self._active_briefs.get(report.brief_id)
        if brief:
            effectiveness = self._evaluate_recommendations(brief, report)
        else:
            effectiveness = {"status": "no_brief_found"}

        # 3. Learn from outcome
        lesson = self._extract_lesson(report)

        # 4. Write learning to Mnemosyne
        self.arsi.store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content=f"[report] {report.agent_id}: {report.task_description[:40]} → {report.outcome} "
                    f"(followed={len(report.recommendations_followed)}, "
                    f"ignored={len(report.recommendations_ignored)})",
            tags=["report", report.agent_id, report.outcome],
            agent_id=report.agent_id,
            generation=self.arsi.store.current_generation,
            importance=report.effect,
        ))

        logger.info(f"Report processed: {report.agent_id} → {report.outcome}")

        return {
            "status": "processed",
            "brief_id": report.brief_id,
            "effectiveness": effectiveness,
            "lesson": lesson,
            "claim_gate": claim_gate,
            "wm_results": wm_results,
            "capture_missing": (report.measurements or {}).get("capture_missing", False),
            "reversion_block": locals().get("bwhy", ""),
            "fail_class": getattr(report, "fail_class", ""),
            "actionable": {
                "recovery_attempted": getattr(report, "recovery_attempted", ""),
                "recovery_worked": getattr(report, "recovery_worked", None),
                "acceptance_evidence": getattr(report, "acceptance_evidence", ""),
            },
        }

    def _goals_from_task(self, task_description: str) -> list[str]:
        """Split task into goal candidates for WM (simple; hosts may override)."""
        text = str(task_description or "").strip()
        if not text:
            return ["complete_task"]
        for sep in ("；", ";", " and then ", " then "):
            if sep in text:
                return [p.strip() for p in text.split(sep) if p.strip()][:8]
        return [text]

    def _generate_warnings(self, agent_id: str) -> list[str]:
        """Generate warnings based on known issues."""
        warnings = []
        traces = self.arsi.store.get_recent_traces(n=50, agent_id=agent_id)

        # Check failure rates by action
        action_outcomes = {}
        for t in traces:
            a = t.get("action", "unknown")
            action_outcomes.setdefault(a, []).append(t.get("outcome", ""))

        for action, outcomes in action_outcomes.items():
            fail_rate = sum(1 for o in outcomes if "fail" in o.lower()) / max(len(outcomes), 1)
            if fail_rate > 0.5:
                warnings.append(f"'{action}' has {fail_rate:.0%} failure rate — be cautious")

        return warnings

    def _generate_recommendations(
        self, task: str, relevant_exp: list, cross_lessons: list
    ) -> list[str]:
        """Generate recommendations from experience."""
        recommendations = []

        # From relevant experiences
        for exp in relevant_exp[:3]:
            if exp.zone == MemoryZone.EXPERIENCE and "empowerment" not in exp.content:
                recommendations.append(f"Past experience: {exp.content[:80]}")

        # From cross-agent lessons
        for lesson in cross_lessons[:2]:
            recommendations.append(f"From {lesson.agent_id}: {lesson.content[:80]}")

        # Default recommendations based on task type
        task_lower = task.lower()
        if any(w in task_lower for w in ["fix", "bug", "debug", "error"]):
            recommendations.append("先复现问题，再定位原因，最后修复验证")
        elif any(w in task_lower for w in ["write", "create", "build"]):
            recommendations.append("先规划结构，再实现，最后测试")
        elif any(w in task_lower for w in ["analyze", "research"]):
            recommendations.append("先收集数据，再分析，最后总结")

        return recommendations[:5]

    def _identify_skills(self, task: str) -> list[str]:
        """Identify relevant skills for the task."""
        skills = []
        task_lower = task.lower()

        skill_map = {
            "arsi-metacognition": ["debug", "fix", "error", "troubleshoot", "bug"],
            "arsi-calibration": ["predict", "estimate", "confidence", "assess"],
            "arsienvironment": ["setup", "config", "install", "deploy"],
            "caveman": ["code", "implement", "write", "build"],
            "github": ["git", "push", "pull", "commit", "repo"],
        }

        for skill_name, keywords in skill_map.items():
            if any(kw in task_lower for kw in keywords):
                skills.append(skill_name)

        return skills

    def _extract_lessons(self, cross_lessons: list) -> list[str]:
        """Extract lessons from cross-agent experiences."""
        lessons = []
        for lesson in cross_lessons[:3]:
            if lesson.agent_id and lesson.agent_id != "unknown":
                lessons.append(f"[{lesson.agent_id}] {lesson.content[:80]}")
        return lessons

    def _build_history_simulator(self, task_description: str) -> dict:
        """Dream-RSI: Build queryable history from discovery tree.

        Key insight: history must be used as an interactive replay
        simulator, not as directionless advice.
        """
        tree = self.arsi.discovery_tree
        if not tree.nodes or len(tree.nodes) < 3:
            return {"available": False, "reason": "insufficient_history"}

        # Find relevant nodes by action/task similarity
        task_lower = task_description.lower()
        relevant = []
        for node in tree.nodes.values():
            if node.action == "start":
                continue
            # Simple relevance: check if task words appear in action
            action_lower = node.action.lower()
            if any(w in action_lower for w in task_lower.split() if len(w) > 2):
                relevant.append(node)

        if len(relevant) < 2:
            # Fallback: use all non-root nodes
            relevant = [n for n in tree.nodes.values() if n.action != "start"]

        if len(relevant) < 2:
            return {"available": False, "reason": "insufficient_relevant_data"}

        successes = [n for n in relevant if n.outcome == "success"]
        failures = [n for n in relevant if n.outcome == "failure"]

        # Extract common failure patterns (structured, not advice)
        fail_patterns = []
        for f in failures[:5]:
            fail_patterns.append(f.action[:40])

        scores = [n.score for n in relevant]
        score_stats = {
            "avg": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "best": round(max(scores), 4) if scores else 0.0,
            "worst": round(min(scores), 4) if scores else 0.0,
        }
        success_rate = len(successes) / len(relevant)

        # §5.1: structured simulator data only — no best_practice / suggested direction
        return {
            "available": True,
            "success_rate": round(success_rate, 3),
            "sample_size": len(relevant),
            "success_count": len(successes),
            "failure_count": len(failures),
            "common_failure_patterns": fail_patterns[:3],
            "score_stats": score_stats,
            "confidence": min(1.0, len(relevant) / 20.0),
        }

    def _evaluate_recommendations(self, brief: ARSIBrief, report: ARSIReport) -> dict:
        """Evaluate how effective ARSI's recommendations were."""
        followed = len(report.recommendations_followed)
        ignored = len(report.recommendations_ignored)
        total = followed + ignored

        if total == 0:
            return {"status": "no_recommendations_tracked"}

        follow_rate = followed / total
        outcome_success = report.outcome == "success"

        # Did following recommendations correlate with success?
        if followed > 0 and outcome_success:
            verdict = "recommendations_seemed_helpful"
        elif ignored > 0 and not outcome_success:
            verdict = "ignoring_recommendations_hurt"
        elif ignored > 0 and outcome_success:
            verdict = "agent_succeeded_without_recommendations"
        else:
            verdict = "unclear"

        return {
            "follow_rate": round(follow_rate, 2),
            "followed": followed,
            "ignored": ignored,
            "outcome": report.outcome,
            "verdict": verdict,
        }

    def _extract_lesson(self, report: ARSIReport) -> str:
        """Extract a lesson from the report."""
        if report.outcome == "success":
            if report.recommendations_followed:
                return f"Following recommendations led to success: {report.task_description[:40]}"
            else:
                return f"Success without following recommendations: {report.task_description[:40]}"
        elif report.outcome == "failure":
            if report.recommendations_ignored:
                return f"Ignoring recommendations led to failure: {report.task_description[:40]}"
            else:
                return f"Failure despite following recommendations: {report.task_description[:40]}"
        return ""

    def _write_feedback(self, brief: ARSIBrief) -> None:
        """Write brief to feedback file — fused write + evidence receipt (B-line)."""
        try:
            FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
            feedback_file = FEEDBACK_DIR / "latest_brief.md"
            body = brief.format_for_agent()
            try:
                from arsi.foundation.evidence_receipt import build_receipt_from_text, fusion_write_and_verify
                result = fusion_write_and_verify(feedback_file, body, receipt_kind="host_brief")
                brief.channel_receipt = result.get("receipt")
                # also persist structured receipt json beside feedback
                build_receipt_from_text(
                    body,
                    source_kind="host_brief",
                    fields={"brief_id": brief.brief_id, "agent_id": brief.agent_id},
                    archive_name=f"host_brief_{brief.brief_id}.txt",
                )
            except Exception as e:
                feedback_file.write_text(body, encoding="utf-8")
                logger.warning(f"receipt path failed, plain write: {e}")
        except Exception as e:
            logger.warning(f"Failed to write feedback: {e}")

    @property
    def stats(self) -> dict:
        return {
            "brief_count": self._brief_count,
            "report_count": self._report_count,
            "active_briefs": len(self._active_briefs),
        }
