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

        from arsi.sealed_eval.code_verifier import CodeVerifier
        verifier = CodeVerifier()
        score, kw_evidence = verifier.verify_keyword_match(output, task.expected_keywords)
        return score, {**evidence, **kw_evidence, "method": "keyword_match_enhanced"}

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
        """For code repair: execute code and verify with assertions."""
        from arsi.sealed_eval.code_verifier import CodeVerifier

        verifier = CodeVerifier(timeout=task.timeout)

        # Parse test cases from expected_behavior
        test_cases = self._parse_test_cases(task.expected_behavior)
        if test_cases:
            score, exec_evidence = verifier.verify_python_code(output, test_cases)
            return score, {**evidence, **exec_evidence, "method": "code_execution"}

        # Fallback: extract code and check syntax + basic structure
        extracted = verifier._extract_code(output)
        if not extracted:
            return 0.0, {**evidence, "reason": "no_code_found"}

        try:
            import ast
            ast.parse(extracted)
            has_function = "def " in extracted
            has_return = "return" in extracted
            score = 0.5 if has_function else 0.2
            if has_return:
                score += 0.2
            return min(score, 1.0), {
                **evidence,
                "method": "syntax_check",
                "has_function": has_function,
                "has_return": has_return,
                "code_length": len(extracted),
            }
        except SyntaxError as e:
            return 0.0, {**evidence, "method": "syntax_check", "syntax_error": str(e)}

    def _parse_test_cases(self, expected_behavior: str) -> list[dict]:
        """Parse test cases from expected_behavior string.

        Format: "func(args)==value and func(args2)==value2"
        or: "func(args)==value; func(args2)==value2"
        """
        if not expected_behavior:
            return []

        test_cases = []
        # Split by "and" or ";"
        parts = re.split(r'\s+and\s+|\s*;\s*', expected_behavior)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Pattern: expr == value
            m = re.match(r'(.+?)\s*==\s*(.+)', part)
            if m:
                expr = m.group(1).strip()
                expected = m.group(2).strip()
                test_cases.append({
                    "assertion": f"assert {expr} == {expected}",
                    "description": part,
                })
            else:
                # Pattern: just an expression (truthy check)
                test_cases.append({
                    "assertion": f"assert {part}",
                    "description": part,
                })

        return test_cases

    def get_history(self) -> list[BaselineReport]:
        return list(self._history)
