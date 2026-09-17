"""LLM-powered enhancements for ARSI core modules.

Wraps existing heuristic modules with LLM intelligence.
Falls back to heuristics when LLM is unavailable.

Usage:
    from arsi.llm_brain import LLMPoweredGovernor, LLMPoweredMindZero, LLMPoweredDiagnosis
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from arsi.foundation.llm import LLMClient
from arsi.foundation.schema import (
    Belief,
    MentalState,
    WorldState,
)

logger = logging.getLogger(__name__)


def _parse_llm_json(response_content: str) -> Optional[dict]:
    """Parse JSON from LLM response, handling markdown fences."""
    content = response_content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


class LLMPoweredMindZero:
    """MindZero with LLM-powered mental state inference.

    Falls back to heuristic frequency-based inference when LLM unavailable.
    """

    def __init__(self, llm: Optional[LLMClient] = None, heuristic_mindzero=None):
        self.llm = llm
        self.heuristic = heuristic_mindzero
        self._llm_calls = 0
        self._llm_failures = 0

    def infer_mental_state(self, traces: list[dict]) -> MentalState:
        """Infer mental state using LLM if available, else heuristic."""
        if self.llm and self.llm.available and traces:
            try:
                result = self._llm_infer(traces)
                if result:
                    self._llm_calls += 1
                    return result
            except Exception as e:
                logger.warning(f"LLM MindZero failed, falling back: {e}")
                self._llm_failures += 1

        # Fallback to heuristic
        if self.heuristic:
            return self.heuristic.infer_mental_state(traces)
        return MentalState()

    def _llm_infer(self, traces: list[dict]) -> Optional[MentalState]:
        """LLM-powered mental state inference."""
        # Summarize traces for the prompt
        trace_summary = []
        for t in traces[-15:]:
            trace_summary.append({
                "action": t.get("action", ""),
                "outcome": t.get("outcome", ""),
                "effect": t.get("effect", 0),
            })

        prompt = f"""分析以下 AI 系统的行为轨迹，推断其心智状态。

行为轨迹（最近15条）：
{json.dumps(trace_summary, ensure_ascii=False)}

输出 JSON：
{{
  "beliefs": [{{"content": "信念描述", "confidence": 0.0到1.0}}],
  "goals": ["目标描述"],
  "affect": {{"管道名": -1.0到1.0}}
}}"""

        response = self.llm.chat(
            prompt,
            system="你是 AI 系统心智状态分析器。只输出 JSON，不要其他文字。",
            max_tokens=800,
        )

        if not response.success:
            return None

        data = _parse_llm_json(response.content)
        if not data:
            return None

        beliefs = []
        for b in data.get("beliefs", []):
            if isinstance(b, dict):
                beliefs.append(Belief(
                    content=b.get("content", ""),
                    confidence=max(0.0, min(1.0, b.get("confidence", 0.5))),
                    source="llm_mindzero",
                ))
            elif isinstance(b, str):
                beliefs.append(Belief(content=b, confidence=0.5, source="llm_mindzero"))

        return MentalState(
            beliefs=beliefs,
            affect=data.get("affect", {}),
        )

    @property
    def stats(self) -> dict:
        return {"llm_calls": self._llm_calls, "llm_failures": self._llm_failures}


class LLMPoweredGovernor:
    """Governor with LLM-assisted decision making.

    Uses LLM to analyze state and select best action,
    falls back to heuristic when LLM unavailable.
    """

    def __init__(self, llm: Optional[LLMClient] = None, heuristic_governor=None):
        self.llm = llm
        self.heuristic = heuristic_governor
        self._llm_calls = 0
        self._llm_failures = 0

    def decide(self, state: WorldState, candidates: list[str]) -> dict:
        """Make a decision using LLM if available, else heuristic."""
        if self.llm and self.llm.available:
            try:
                result = self._llm_decide(state, candidates)
                if result:
                    self._llm_calls += 1
                    return result
            except Exception as e:
                logger.warning(f"LLM Governor failed, falling back: {e}")
                self._llm_failures += 1

        # Fallback
        return {
            "action": candidates[0] if candidates else "remember",
            "reason": "heuristic_fallback",
            "expected_effect": "",
            "risk": "low",
        }

    def _llm_decide(self, state: WorldState, candidates: list[str]) -> Optional[dict]:
        """LLM-powered decision."""
        prompt = f"""你是自创生 AI 系统的 Governor。根据当前状态选择下一步动作。

系统状态：
- 代数: {state.phi.generation}
- η（自我模型失配度）: {state.eta:.3f}
- 存储: {json.dumps(state.phi.storage_stats, ensure_ascii=False)}
- 信念数: {len(state.psi.beliefs)}
- 情绪: {json.dumps(state.psi.affect, ensure_ascii=False)}

候选动作: {candidates}

选择一个动作并说明理由。输出 JSON：
{{"action": "选择的动作", "reason": "理由", "expected_effect": "预期效果", "risk": "low|medium|high"}}"""

        response = self.llm.chat(
            prompt,
            system="你是理性的系统调度器。只输出 JSON。",
            max_tokens=500,
        )

        if not response.success:
            return None

        data = _parse_llm_json(response.content)
        if not data or "action" not in data:
            return None

        # Validate action is in candidates
        if data["action"] not in candidates:
            data["action"] = candidates[0] if candidates else "remember"

        return {
            "action": data["action"],
            "reason": data.get("reason", ""),
            "expected_effect": data.get("expected_effect", ""),
            "risk": data.get("risk", "low"),
        }

    @property
    def stats(self) -> dict:
        return {"llm_calls": self._llm_calls, "llm_failures": self._llm_failures}


class LLMPoweredDiagnosis:
    """Empowerment diagnosis with LLM analysis."""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm
        self._llm_calls = 0

    def diagnose(self, traces: list[dict], state: WorldState) -> dict:
        """Diagnose agent weakness using LLM."""
        if not self.llm or not self.llm.available or not traces:
            return {"dimension": "knowledge", "reason": "llm_unavailable_or_no_traces"}

        trace_summary = []
        for t in traces[-20:]:
            trace_summary.append({
                "action": t.get("action", ""),
                "outcome": t.get("outcome", ""),
                "effect": t.get("effect", 0),
            })

        prompt = f"""分析以下 agent 行为轨迹，找出最需要改进的赋能维度。

行为轨迹：
{json.dumps(trace_summary, ensure_ascii=False)}

九个维度：skill, harness, knowledge, decomposition, metacognition, attention, environment, social_graph, calibration

输出 JSON：
{{"dimension": "维度名", "reason": "诊断依据", "recommendation": "改进建议", "confidence": 0.0到1.0}}"""

        response = self.llm.chat(
            prompt,
            system="你是精确的诊断专家。只输出 JSON。",
            max_tokens=600,
        )

        if not response.success:
            return {"dimension": "knowledge", "reason": "llm_call_failed"}

        data = _parse_llm_json(response.content)
        if not data:
            return {"dimension": "knowledge", "reason": "llm_parse_failed"}

        self._llm_calls += 1
        valid_dims = {"skill", "harness", "knowledge", "decomposition",
                       "metacognition", "attention", "environment", "social_graph", "calibration"}
        if data.get("dimension") not in valid_dims:
            data["dimension"] = "knowledge"

        return data

    @property
    def stats(self) -> dict:
        return {"llm_calls": self._llm_calls}


class LLMPoweredDream:
    """Dream pipeline with LLM-powered belief reconciliation."""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm
        self._llm_calls = 0

    def reconcile_beliefs(self, old_beliefs: list[Belief], new_beliefs: list[Belief]) -> list[Belief]:
        """Reconcile old and new beliefs using LLM semantic comparison."""
        if not self.llm or not self.llm.available:
            # Heuristic: prefer new, keep high-confidence old
            reconciled = list(new_beliefs)
            for ob in old_beliefs:
                if ob.confidence > 0.7:
                    reconciled.append(ob)
            return reconciled

        old_str = json.dumps([{"content": b.content, "confidence": b.confidence} for b in old_beliefs[-10:]], ensure_ascii=False)
        new_str = json.dumps([{"content": b.content, "confidence": b.confidence} for b in new_beliefs[-10:]], ensure_ascii=False)

        prompt = f"""调和 AI 系统的新旧信念。保留准确的，修正过时的。

旧信念：
{old_str}

新信念（从最近行为推断）：
{new_str}

输出 JSON：
{{"reconciled": [{{"content": "调和后的信念", "confidence": 0.0到1.0, "source": "old|new|merged"}}]}}"""

        response = self.llm.chat(
            prompt,
            system="你是信念调和专家。只输出 JSON。",
            max_tokens=800,
        )

        if not response.success:
            return list(new_beliefs) + [b for b in old_beliefs if b.confidence > 0.7]

        data = _parse_llm_json(response.content)
        if not data or "reconciled" not in data:
            return list(new_beliefs)

        self._llm_calls += 1
        reconciled = []
        for b in data["reconciled"]:
            if isinstance(b, dict) and "content" in b:
                reconciled.append(Belief(
                    content=b["content"],
                    confidence=max(0.0, min(1.0, b.get("confidence", 0.5))),
                    source=f"llm_reconciled:{b.get('source', 'unknown')}",
                ))
        return reconciled

    @property
    def stats(self) -> dict:
        return {"llm_calls": self._llm_calls}
