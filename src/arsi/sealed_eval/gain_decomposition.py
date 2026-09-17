"""Gain Decomposition — amplify / import / self-organize.

Decomposes capability changes into three sources:
  amplified:      existing capability improved (effect routing, param tuning)
  imported:       new capability from external source (papers, knowledge injection)
  self_organized: structural reorganization (pipeline reordering, memory consolidation)

Based on: ARSI Whitepaper §7.5 + MetaRSI Law 5
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from arsi.foundation.llm import LLMClient
from arsi.foundation.store import MnemosyneStore

logger = logging.getLogger(__name__)


def _parse_llm_json(content: str) -> Optional[dict]:
    c = content.strip()
    if c.startswith("```json"):
        c = c[7:]
    if c.startswith("```"):
        c = c[3:]
    if c.endswith("```"):
        c = c[:-3]
    c = c.strip()
    try:
        return json.loads(c)
    except json.JSONDecodeError:
        start = c.find("{")
        end = c.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(c[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


class GainDecomposer:
    """Decomposes term gains into amplified / imported / self_organized.

    Uses trace analysis + LLM to attribute gains to their sources.
    """

    def __init__(self, store: MnemosyneStore, llm: Optional[LLMClient] = None):
        self.store = store
        self.llm = llm
        self._decomposition_count = 0

    def decompose(
        self,
        before_eval: dict,
        after_eval: dict,
        traces: list[dict],
    ) -> dict:
        """Decompose gain into three sources.

        Args:
            before_eval: evaluation before term (capability, per_task)
            after_eval: evaluation after term
            traces: behavior traces from the term

        Returns:
            {amplified: float, imported: float, self_organized: float,
             total: float, evidence: dict}
        """
        self._decomposition_count += 1

        total_gain = after_eval.get("capability", 0) - before_eval.get("capability", 0)

        if total_gain <= 0:
            return {
                "amplified": 0.0,
                "imported": 0.0,
                "self_organized": 0.0,
                "total": round(total_gain, 4),
                "evidence": {"reason": "no_gain"},
            }

        # Rule-based attribution from traces
        rule_result = self._rule_based_attribution(traces, total_gain)

        # LLM-enhanced attribution if available
        if self.llm and self.llm.available:
            llm_result = self._llm_attribution(traces, total_gain)
            if llm_result:
                # Blend: LLM provides proportions, rules provide grounding
                return self._blend(rule_result, llm_result, total_gain)

        return rule_result

    def _rule_based_attribution(self, traces: list[dict], total_gain: float) -> dict:
        """Attribute gains based on action patterns in traces."""
        action_counts = {}
        for t in traces:
            a = t.get("action", "unknown")
            action_counts[a] = action_counts.get(a, 0) + 1

        total_actions = sum(action_counts.values()) or 1

        # Classify actions into gain types
        # amplified: evolve, reflect (improving existing)
        # imported: learn (bringing in new knowledge)
        # self_organized: dream, maintain (restructuring)
        amplified_actions = action_counts.get("evolve", 0) + action_counts.get("reflect", 0)
        imported_actions = action_counts.get("learn", 0)
        self_org_actions = action_counts.get("dream", 0) + action_counts.get("maintain", 0)

        # Proportional attribution
        total_classified = amplified_actions + imported_actions + self_org_actions
        if total_classified == 0:
            # Default: all amplified
            return {
                "amplified": round(total_gain, 4),
                "imported": 0.0,
                "self_organized": 0.0,
                "total": round(total_gain, 4),
                "evidence": {"method": "rule_default", "reason": "no_classifiable_actions"},
            }

        amplified_ratio = amplified_actions / total_classified
        imported_ratio = imported_actions / total_classified
        self_org_ratio = self_org_actions / total_classified

        return {
            "amplified": round(total_gain * amplified_ratio, 4),
            "imported": round(total_gain * imported_ratio, 4),
            "self_organized": round(total_gain * self_org_ratio, 4),
            "total": round(total_gain, 4),
            "evidence": {
                "method": "rule_based",
                "action_counts": action_counts,
                "ratios": {
                    "amplified": round(amplified_ratio, 3),
                    "imported": round(imported_ratio, 3),
                    "self_organized": round(self_org_ratio, 3),
                },
            },
        }

    def _llm_attribution(self, traces: list[dict], total_gain: float) -> Optional[dict]:
        """Use LLM to attribute gains more intelligently."""
        trace_summary = json.dumps(
            [{"action": t.get("action"), "outcome": t.get("outcome"),
              "effect": t.get("effect", 0)} for t in traces[-20:]],
            ensure_ascii=False,
        )

        prompt = f"""分析以下行为轨迹，将能力提升 {total_gain:.3f} 分解为三个来源。

轨迹：
{trace_summary}

三个来源定义：
- amplified（放大）：已有能力的改进，如效果路由优化、参数调整
- imported（进口）：从外部搬入的新能力，如学习新知识、移植新机制
- self_organized（自组织）：系统重组自身结构带来的改进，如记忆整合、管道重排

输出 JSON：
{{"amplified": 0.0到1.0的比例, "imported": 0.0到1.0的比例, "self_organized": 0.0到1.0的比例, "reasoning": "推理过程"}}"""

        resp = self.llm.chat(prompt, system="你是增益归因专家。只输出 JSON。", max_tokens=500)
        if not resp.success:
            return None

        data = _parse_llm_json(resp.content)
        if not data:
            return None

        # Normalize proportions
        total_ratio = (data.get("amplified", 0) + data.get("imported", 0) +
                       data.get("self_organized", 0))
        if total_ratio <= 0:
            return None

        return {
            "amplified_ratio": data.get("amplified", 0) / total_ratio,
            "imported_ratio": data.get("imported", 0) / total_ratio,
            "self_organized_ratio": data.get("self_organized", 0) / total_ratio,
            "reasoning": data.get("reasoning", ""),
        }

    def _blend(self, rule_result: dict, llm_result: dict, total_gain: float) -> dict:
        """Blend rule-based and LLM attributions."""
        # Weight: 40% rules, 60% LLM (LLM is smarter but rules are grounded)
        w_rule = 0.4
        w_llm = 0.6

        amplified = (rule_result.get("amplified", 0) * w_rule +
                     total_gain * llm_result.get("amplified_ratio", 0) * w_llm)
        imported = (rule_result.get("imported", 0) * w_rule +
                    total_gain * llm_result.get("imported_ratio", 0) * w_llm)
        self_org = (rule_result.get("self_organized", 0) * w_rule +
                    total_gain * llm_result.get("self_organized_ratio", 0) * w_llm)

        return {
            "amplified": round(amplified, 4),
            "imported": round(imported, 4),
            "self_organized": round(self_org, 4),
            "total": round(total_gain, 4),
            "evidence": {
                "method": "blended",
                "rule_result": rule_result.get("evidence", {}),
                "llm_reasoning": llm_result.get("reasoning", ""),
            },
        }

    def record(self, decomposition: dict, term_id: str) -> None:
        """Record decomposition to store."""
        self.store.write_memory(
            __import__("arsi.foundation.schema", fromlist=["MemoryRecord"]).MemoryRecord(
                zone=__import__("arsi.foundation.schema", fromlist=["MemoryZone"]).MemoryZone.SELF,
                content=json.dumps({
                    "term_id": term_id,
                    "amplified": decomposition["amplified"],
                    "imported": decomposition["imported"],
                    "self_organized": decomposition["self_organized"],
                    "total": decomposition["total"],
                }, ensure_ascii=False),
                tags=["gain_decomposition", term_id],
                importance=0.9,
            )
        )

    @property
    def stats(self) -> dict:
        return {"decomposition_count": self._decomposition_count}
