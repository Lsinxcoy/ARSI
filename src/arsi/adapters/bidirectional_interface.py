"""ARSI Bidirectional Interface Protocol — agents pass through ARSI on every task.

Protocol:
  1. Agent calls brief() before task → gets context, skills, recommendations
  2. Agent executes task
  3. Agent calls report() after task → ARSI learns from outcome

ARSI only improves its own consulting quality. Agents don't need to change
their core behavior — they just call ARSI before/after.
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

    brief_policy (Dream-RSI §5.1):
      - structured (default): skills + safety warnings + structured history
        stats only. Recommendations / past lessons are meta_only and MUST NOT
        appear in format_for_agent() — semantic guidance hurts exploration.
      - full: legacy text brief for human CLI / debugging.
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
            "timestamp": self.timestamp,
        }

    def format_for_agent(self) -> str:
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
    """What an agent tells ARSI after a task."""

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
        self.timestamp = datetime.now().isoformat()


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

    def brief(self, task_description: str, agent_id: str) -> ARSIBrief:
        """Generate a brief for an agent about to execute a task.

        This is what ARSI "says" to the agent before it works.
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

        # 1. Record the trace
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
        }

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
