"""Trace Quality Gate — NeoHorse-inspired three-gate pipeline.

Gate 1: Structural validation (fields present, effect in range)
Gate 2: Semantic evaluation (six dimensions, LLM-powered)
Gate 3: Difficulty classification (C0-C3, for curriculum scheduling)

Based on: NeoHorse-1 (arXiv:2609.08183)
Key finding: real production traffic > synthetic data (6.26 points)
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from arsi.foundation.llm import LLMClient

logger = logging.getLogger(__name__)


class DifficultyLevel(str, Enum):
    C0 = "C0"  # Simple/repetitive
    C1 = "C1"  # Standard
    C2 = "C2"  # Complex/multi-step
    C3 = "C3"  # Novel/ambiguous


class QualityVerdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    NOT_EVALUATED = "NOT_EVALUATED"


class TraceQualityGate:
    """NeoHorse-inspired quality pipeline for trace ingestion."""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm
        self._gate1_pass = 0
        self._gate1_fail = 0
        self._gate3_counts = {"C0": 0, "C1": 0, "C2": 0, "C3": 0}

    def gate1_structural(self, trace: dict) -> tuple[bool, str]:
        """Structural validation: required fields + valid ranges."""
        if not trace.get("action"):
            self._gate1_fail += 1
            return False, "missing_action"
        if not trace.get("outcome"):
            self._gate1_fail += 1
            return False, "missing_outcome"

        effect = trace.get("effect", None)
        if effect is None:
            self._gate1_fail += 1
            return False, "missing_effect"
        if not (-1 <= effect <= 1):
            self._gate1_fail += 1
            return False, f"effect_out_of_range:{effect}"

        self._gate1_pass += 1
        return True, "ok"

    def gate2_semantic(self, trace: dict) -> dict:
        """Semantic evaluation: six dimensions.

        NeoHorse dimensions:
        - goal_achievement: did the task achieve its goal?
        - instruction_following: did it follow instructions?
        - tool_usage: were tools used appropriately?
        - evidence_consistency: is evidence consistent?
        - error_recovery: did it recover from errors?
        - termination: did it terminate properly?
        """
        if not self.llm or not self.llm.available:
            return self._heuristic_semantic(trace)

        prompt = f"""评估以下行为轨迹的质量：

动作: {trace.get('action', '?')}
结果: {trace.get('outcome', '?')}
效果: {trace.get('effect', 0)}
参数: {trace.get('params', {})}

六个维度各给 PASS/WARN/FAIL：
1. goal_achievement — 目标达成
2. instruction_following — 指令遵循
3. tool_usage — 工具使用
4. evidence_consistency — 证据一致性
5. error_recovery — 错误恢复
6. termination — 终止恰当性

输出 JSON：
{{"goal_achievement": "PASS", "instruction_following": "PASS", "tool_usage": "WARN", "evidence_consistency": "PASS", "error_recovery": "N/A", "termination": "PASS"}}"""

        resp = self.llm.chat(prompt, system="你是轨迹质量评估器。只输出 JSON。", max_tokens=200)
        if resp.success:
            import json
            try:
                content = resp.content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                return json.loads(content)
            except Exception:
                pass
        return self._heuristic_semantic(trace)

    def _heuristic_semantic(self, trace: dict) -> dict:
        """Heuristic fallback for semantic evaluation."""
        outcome = trace.get("outcome", "")
        effect = trace.get("effect", 0.5)

        if outcome == "success" and effect > 0.7:
            verdict = QualityVerdict.PASS
        elif outcome == "failure":
            verdict = QualityVerdict.FAIL
        else:
            verdict = QualityVerdict.WARN

        return {
            "goal_achievement": verdict.value,
            "instruction_following": QualityVerdict.NOT_EVALUATED.value,
            "tool_usage": QualityVerdict.NOT_EVALUATED.value,
            "evidence_consistency": QualityVerdict.NOT_EVALUATED.value,
            "error_recovery": QualityVerdict.NOT_EVALUATED.value,
            "termination": QualityVerdict.NOT_EVALUATED.value,
        }

    def gate3_difficulty(self, trace: dict) -> DifficultyLevel:
        """Classify task difficulty: C0-C3.

        Based on outcome, effect, and action complexity.
        """
        action = trace.get("action", "")
        effect = trace.get("effect", 0.5)
        outcome = trace.get("outcome", "")
        params = trace.get("params", {})
        param_complexity = len(str(params))

        # C3: Novel/ambiguous (high param complexity + failure)
        if outcome == "failure" and param_complexity > 200:
            level = DifficultyLevel.C3
        # C2: Complex/multi-step (low effect or high failure)
        elif effect < 0.4 or outcome == "failure":
            level = DifficultyLevel.C2
        # C1: Standard (good outcome)
        elif outcome == "success" and effect > 0.7:
            level = DifficultyLevel.C1
        # C0: Simple/repetitive
        else:
            level = DifficultyLevel.C0

        self._gate3_counts[level.value] = self._gate3_counts.get(level.value, 0) + 1
        return level

    def evaluate(self, trace: dict) -> dict:
        """Run all three gates on a trace."""
        # Gate 1: Structural
        g1_pass, g1_reason = self.gate1_structural(trace)
        if not g1_pass:
            return {"accepted": False, "gate1": g1_reason, "difficulty": None}

        # Gate 2: Semantic
        g2_result = self.gate2_semantic(trace)

        # Gate 3: Difficulty
        g3_level = self.gate3_difficulty(trace)

        return {
            "accepted": True,
            "gate1": "ok",
            "gate2": g2_result,
            "gate3_difficulty": g3_level.value,
        }

    @property
    def stats(self) -> dict:
        return {
            "gate1_pass": self._gate1_pass,
            "gate1_fail": self._gate1_fail,
            "gate3_distribution": self._gate3_counts,
        }


class CurriculumScheduler:
    """NeoHorse-inspired curriculum: learn from easy first, then hard."""

    def __init__(self):
        self._current_term = 0

    def set_term(self, term: int):
        self._current_term = term

    def select_traces(self, traces: list, quality_results: list) -> list:
        """Select traces based on curriculum schedule."""
        # Map difficulty to curriculum phases
        if self._current_term <= 2:
            allowed = {"C0", "C1"}
        elif self._current_term <= 4:
            allowed = {"C0", "C1", "C2"}
        else:
            allowed = {"C0", "C1", "C2", "C3"}

        selected = []
        for trace, quality in zip(traces, quality_results):
            if not quality.get("accepted", False):
                continue
            difficulty = quality.get("gate3_difficulty", "C0")
            if difficulty in allowed:
                trace["_difficulty"] = difficulty
                selected.append(trace)

        return selected

    @property
    def phase_description(self) -> str:
        if self._current_term <= 2:
            return "Phase 1: C0-C1 (simple tasks)"
        elif self._current_term <= 4:
            return "Phase 2: C0-C2 (standard + complex)"
        else:
            return "Phase 3: C0-C3 (full range)"
