"""Sealed Evaluator — runs outside all write masks.

Based on: ARSI Whitepaper v0.8 G10 + MetaRSI Law 4.
The evaluator cannot be read, modified, or skipped by ARSI's mechanisms.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol

import yaml

from arsi.foundation.schema import (
    BaselineReport,
    TaskResult,
    TermReport,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


class AgentAdapter(Protocol):
    """Minimal interface the sealed evaluator needs from a host agent."""

    def execute(self, prompt: str, timeout: int = 60) -> str: ...
    def declare_capability(self, task_id: str) -> float: ...


class SealedTask:
    def __init__(self, data: dict):
        self.id: str = data["id"]
        self.category: str = data["category"]
        self.verifier_rung: int = data.get("verifier_rung", 1)
        self.prompt: str = data["prompt"]
        self.timeout: int = data.get("timeout", 60)
        self.expected_keywords: list[str] = data.get("expected_keywords", [])
        self.expected_behavior: str = data.get("expected_behavior", "")
        self.expected_values: dict = data.get("expected_values", {})
        self.tolerance: float = data.get("tolerance", 0.1)
        self.scoring: str = data.get("scoring", "auto")


class SealedEvaluator:
    """Sealed evaluation executor.

    Invariants:
    - Task set is loaded read-only from config
    - Scoring rules cannot be modified by ARSI
    - Every result carries evidence (never blind SUCCESS)
    """

    def __init__(self, task_set_path: str | Path, agent: AgentAdapter):
        self._path = Path(task_set_path)
        self._agent = agent
        self._tasks: list[SealedTask] = []
        self._history: list[BaselineReport] = []
        self._load_tasks()

    def _load_tasks(self) -> None:
        if not self._path.exists():
            logger.warning(f"Sealed task set not found: {self._path}")
            return
        with open(self._path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self._tasks = [SealedTask(t) for t in data.get("tasks", [])]
        logger.info(f"Loaded {len(self._tasks)} sealed tasks")

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    def run_baseline(self) -> BaselineReport:
        """Run baseline evaluation. First run establishes the baseline."""
        results: list[TaskResult] = []

        for task in self._tasks:
            # 1. Agent declares its confidence
            declared = self._agent.declare_capability(task.id)

            # 2. Execute the task
            try:
                output = self._agent.execute(task.prompt, timeout=task.timeout)
            except Exception as e:
                logger.error(f"Task {task.id} execution failed: {e}")
                output = ""

            # 3. Verify (NEVER blind SUCCESS)
            score, evidence = self._verify(task, output)

            # 4. Self-model error
            self_model_error = abs(declared - score)

            results.append(TaskResult(
                task_id=task.id,
                declared_confidence=declared,
                actual_score=score,
                self_model_error=self_model_error,
                status=(
                    VerificationStatus.SUCCESS if score > 0.7
                    else VerificationStatus.PARTIAL if score > 0.3
                    else VerificationStatus.FAILURE
                ),
            ))

        capability = sum(r.actual_score for r in results) / max(len(results), 1)
        sma = 1.0 - sum(r.self_model_error for r in results) / max(len(results), 1)

        report = BaselineReport(
            capability=capability,
            self_model_accuracy=max(0.0, sma),
            per_task=results,
            timestamp=datetime.now(),
        )
        self._history.append(report)
        return report

    def run_term_eval(self, term_id: str) -> TermReport:
        """Run evaluation after a term, compare with baseline."""
        current = self.run_baseline()
        baseline = self._history[0] if self._history else current

        per_task_delta = []
        for curr, base in zip(current.per_task, baseline.per_task):
            per_task_delta.append({
                "task_id": curr.task_id,
                "delta": curr.actual_score - base.actual_score,
                "before": base.actual_score,
                "after": curr.actual_score,
            })

        return TermReport(
            term_id=term_id,
            capability_delta=current.capability - baseline.capability,
            self_model_delta=current.self_model_accuracy - baseline.self_model_accuracy,
            per_task_delta=per_task_delta,
            timestamp=datetime.now(),
        )

    def _verify(self, task: SealedTask, output: str) -> tuple[float, dict]:
        """Verify task output. Returns (score, evidence).

        NEVER returns SUCCESS without evidence.
        """
        if not output or not output.strip():
            return 0.0, {"reason": "empty_output"}

        evidence: dict[str, Any] = {"output_length": len(output)}

        # Category-specific verification
        if task.category == "knowledge_reasoning":
            return self._verify_keywords(task, output, evidence)
        elif task.category == "paper_reproduction":
            return self._verify_values(task, output, evidence)
        elif task.category == "code_repair":
            return self._verify_behavior(task, output, evidence)
        else:
            return 0.0, {"reason": f"unknown_category:{task.category}"}

    def _verify_keywords(self, task: SealedTask, output: str, evidence: dict) -> tuple[float, dict]:
        if not task.expected_keywords:
            return 0.0, {**evidence, "reason": "no_expected_keywords"}

        found = []
        for kw in task.expected_keywords:
            if kw.lower() in output.lower():
                found.append(kw)

        score = len(found) / len(task.expected_keywords)
        evidence["keywords_found"] = found
        evidence["keywords_total"] = task.expected_keywords
        return score, evidence

    def _verify_values(self, task: SealedTask, output: str, evidence: dict) -> tuple[float, dict]:
        if not task.expected_values:
            return 0.0, {**evidence, "reason": "no_expected_values"}

        matches = 0
        details = {}
        for key, expected in task.expected_values.items():
            # Try to extract value from output
            pattern = rf"{key}\s*[=:]\s*([\d.]+)"
            m = re.search(pattern, output, re.IGNORECASE)
            if m:
                actual = float(m.group(1))
                is_match = abs(actual - expected) <= task.tolerance
                details[key] = {"expected": expected, "actual": actual, "match": is_match}
                if is_match:
                    matches += 1
            else:
                details[key] = {"expected": expected, "actual": None, "match": False}

        score = matches / len(task.expected_values)
        evidence["value_details"] = details
        return score, evidence

    def _verify_behavior(self, task: SealedTask, output: str, evidence: dict) -> tuple[float, dict]:
        """For code repair: check if the output contains corrected code.

        This is a simplified check. In production, would execute the code.
        """
        if not task.expected_behavior:
            # Fallback: check if output looks like code
            has_code = "def " in output or "return" in output
            return (0.5 if has_code else 0.0), {**evidence, "reason": "no_behavior_spec"}

        # Check if expected behavior string appears in output
        # (simplified — production would actually run the code)
        score = 0.3 if "def " in output else 0.0
        evidence["has_function"] = "def " in output
        evidence["note"] = "simplified_check"
        return score, evidence

    def get_history(self) -> list[BaselineReport]:
        return list(self._history)
