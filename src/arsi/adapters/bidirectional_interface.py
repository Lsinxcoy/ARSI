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
    """What ARSI tells an agent before a task."""

    def __init__(
        self,
        task_description: str,
        agent_id: str,
        recommendations: list[str],
        relevant_skills: list[str],
        warnings: list[str],
        past_lessons: list[str],
        confidence: float = 0.5,
    ):
        self.task_description = task_description
        self.agent_id = agent_id
        self.recommendations = recommendations
        self.relevant_skills = relevant_skills
        self.warnings = warnings
        self.past_lessons = past_lessons
        self.confidence = confidence
        self.timestamp = datetime.now().isoformat()
        self.brief_id = f"brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def to_dict(self) -> dict:
        return {
            "brief_id": self.brief_id,
            "agent_id": self.agent_id,
            "task": self.task_description,
            "recommendations": self.recommendations,
            "relevant_skills": self.relevant_skills,
            "warnings": self.warnings,
            "past_lessons": self.past_lessons,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    def format_for_agent(self) -> str:
        """Format brief as readable text for agent consumption."""
        lines = [
            f"# ARSI Brief [{self.brief_id}]",
            f"Agent: {self.agent_id}  Task: {self.task_description}",
            f"Confidence: {self.confidence:.2f}",
            "",
        ]
        if self.recommendations:
            lines.append("## Recommendations")
            for r in self.recommendations:
                lines.append(f"  - {r}")
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
        if self.past_lessons:
            lines.append("## Past Lessons")
            for l in self.past_lessons:
                lines.append(f"  - {l}")
        return "\n".join(lines)


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

    def __init__(self, arsi: ARSI):
        self.arsi = arsi
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

        brief = ARSIBrief(
            task_description=task_description,
            agent_id=agent_id,
            recommendations=recommendations,
            relevant_skills=skills,
            warnings=warnings,
            past_lessons=lessons,
            confidence=confidence,
        )

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
        """Write brief to feedback file for agent to read."""
        try:
            FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
            feedback_file = FEEDBACK_DIR / "latest_brief.md"
            feedback_file.write_text(brief.format_for_agent(), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to write feedback: {e}")

    @property
    def stats(self) -> dict:
        return {
            "brief_count": self._brief_count,
            "report_count": self._report_count,
            "active_briefs": len(self._active_briefs),
        }
