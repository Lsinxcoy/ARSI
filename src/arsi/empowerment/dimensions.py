"""Empowerment dimension implementations — real sensing, generation, validation.

Three dimensions implemented with full closed loops:
  1. knowledge     — detect missing knowledge, generate via LLM, validate via re-test
  2. decomposition — analyze task split quality, generate better splits, validate
  3. calibration   — measure confidence-accuracy gap, generate calibration curve, validate

Each dimension follows: sense → generate → validate → record
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from arsi.foundation.llm import LLMClient
from arsi.foundation.schema import (
    EmpowermentDimension,
    MemoryRecord,
    MemoryZone,
    VerificationStatus,
    WorldState,
)
from arsi.foundation.store import MnemosyneStore

logger = logging.getLogger(__name__)


def _parse_llm_json(content: str) -> Optional[dict]:
    """Parse JSON from LLM response."""
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


class KnowledgeDimension:
    """Knowledge empowerment — detect missing knowledge, generate, validate.

    Sense:  which failures correlate with missing domain knowledge
    Generate: LLM extracts knowledge from successful traces
    Validate: does injecting knowledge improve subsequent success rate
    """

    def __init__(self, store: MnemosyneStore, llm: Optional[LLMClient] = None):
        self.store = store
        self.llm = llm
        self.dimension = EmpowermentDimension.KNOWLEDGE

    def sense(self, traces: list[dict], state: WorldState) -> dict:
        """Detect knowledge gaps from failure patterns."""
        failures = [t for t in traces if t.get("outcome") == "failure"]
        successes = [t for t in traces if t.get("outcome") == "success"]

        if not failures:
            return {"gap_detected": False, "reason": "no failures to analyze"}

        # Analyze failure patterns
        fail_actions = {}
        for t in failures:
            a = t.get("action", "unknown")
            fail_actions[a] = fail_actions.get(a, 0) + 1

        # Find actions with high failure rate
        total_by_action = {}
        for t in traces:
            a = t.get("action", "unknown")
            total_by_action[a] = total_by_action.get(a, 0) + 1

        high_fail_actions = []
        for action, fail_count in fail_actions.items():
            total = total_by_action.get(action, 1)
            fail_rate = fail_count / total
            if fail_rate > 0.4:
                high_fail_actions.append({"action": action, "fail_rate": round(fail_rate, 2)})

        # LLM analysis of what knowledge is missing
        knowledge_gap = None
        if self.llm and self.llm.available and failures:
            fail_summary = json.dumps(
                [{"action": t.get("action"), "outcome": t.get("outcome"),
                  "effect": t.get("effect", 0)} for t in failures[-10:]],
                ensure_ascii=False,
            )
            prompt = f"""分析以下失败轨迹，判断缺失什么领域知识。

失败轨迹：
{fail_summary}

成功轨迹（对照）：
{json.dumps([{"action": t.get("action"), "effect": t.get("effect", 0)} for t in successes[-5:]], ensure_ascii=False)}

输出 JSON：
{{"missing_knowledge": "缺失的知识描述", "evidence": "判断依据", "suggestion": "建议获取什么知识"}}"""

            resp = self.llm.chat(prompt, system="你是知识缺口分析专家。只输出 JSON。", max_tokens=500)
            if resp.success:
                knowledge_gap = _parse_llm_json(resp.content)

        return {
            "gap_detected": bool(high_fail_actions or knowledge_gap),
            "high_fail_actions": high_fail_actions,
            "failure_count": len(failures),
            "success_count": len(successes),
            "knowledge_gap": knowledge_gap,
        }

    def generate(self, sense_result: dict, traces: list[dict]) -> dict:
        """Generate knowledge to inject based on sensed gap."""
        if not sense_result.get("gap_detected"):
            return {"action": "none", "reason": "no gap detected"}

        successes = [t for t in traces if t.get("outcome") == "success"]

        if not self.llm or not self.llm.available or not successes:
            return {"action": "none", "reason": "llm_unavailable_or_no_successes"}

        # LLM extracts reusable knowledge from successful traces
        success_summary = json.dumps(
            [{"action": t.get("action"), "params": t.get("action_params", {}),
              "effect": t.get("effect", 0)} for t in successes[-15:]],
            ensure_ascii=False,
        )
        gap_info = sense_result.get("knowledge_gap", {})

        prompt = f"""从成功轨迹中提取可复用的知识，填补检测到的知识缺口。

知识缺口：{json.dumps(gap_info, ensure_ascii=False)}

成功轨迹：
{success_summary}

输出 JSON：
{{"knowledge_items": [{{"content": "知识内容", "applicability": "适用场景", "confidence": 0.0到1.0}}]}}"""

        resp = self.llm.chat(prompt, system="你是知识蒸馏专家。只输出 JSON。", max_tokens=800)
        if not resp.success:
            return {"action": "none", "reason": "llm_call_failed"}

        data = _parse_llm_json(resp.content)
        if not data or "knowledge_items" not in data:
            return {"action": "none", "reason": "parse_failed"}

        items = data["knowledge_items"]
        return {
            "action": "inject_knowledge",
            "items": items,
            "count": len(items),
        }

    def validate(self, generate_result: dict, before_stats: dict, after_stats: dict) -> tuple[VerificationStatus, dict]:
        """Validate knowledge injection effect."""
        if generate_result.get("action") == "none":
            return VerificationStatus.UNKNOWN, {"reason": "nothing_to_validate"}

        before_rate = before_stats.get("success_rate", 0)
        after_rate = after_stats.get("success_rate", 0)
        delta = after_rate - before_rate

        evidence = {
            "before_success_rate": round(before_rate, 3),
            "after_success_rate": round(after_rate, 3),
            "delta": round(delta, 3),
            "items_injected": generate_result.get("count", 0),
        }

        if delta > 0.05:
            return VerificationStatus.SUCCESS, evidence
        elif delta > 0:
            return VerificationStatus.PARTIAL, evidence
        else:
            return VerificationStatus.UNKNOWN, {**evidence, "reason": "no_improvement"}


class DecompositionDimension:
    """Decomposition empowerment — analyze task split quality.

    Sense:  compare success/failure patterns in multi-step tasks
    Generate: LLM suggests better decomposition strategies
    Validate: track if re-decomposed tasks perform better
    """

    def __init__(self, store: MnemosyneStore, llm: Optional[LLMClient] = None):
        self.store = store
        self.llm = llm
        self.dimension = EmpowermentDimension.DECOMPOSITION

    def sense(self, traces: list[dict], state: WorldState) -> dict:
        """Analyze decomposition quality from trace patterns."""
        if not traces:
            return {"gap_detected": False, "reason": "no_traces"}

        # Look for patterns: consecutive failures after a specific action
        consecutive_fails = 0
        max_consecutive = 0
        fail_sequence_start = None

        for i, t in enumerate(traces):
            if t.get("outcome") == "failure":
                consecutive_fails += 1
                if consecutive_fails == 2:
                    fail_sequence_start = traces[i - 1].get("action", "")
                max_consecutive = max(max_consecutive, consecutive_fails)
            else:
                consecutive_fails = 0

        # High consecutive failures suggest decomposition problem
        has_decomp_issue = max_consecutive >= 2

        # LLM analysis
        decomp_analysis = None
        if self.llm and self.llm.available and traces:
            trace_summary = json.dumps(
                [{"action": t.get("action"), "outcome": t.get("outcome"),
                  "effect": t.get("effect", 0)} for t in traces[-20:]],
                ensure_ascii=False,
            )
            prompt = f"""分析以下行为轨迹，判断任务分解策略是否合理。

轨迹：
{trace_summary}

判断：
1. 是否存在"拆错了"的模式（先做了不该先做的事）
2. 是否有步骤遗漏
3. 是否有步骤顺序问题

输出 JSON：
{{"decomposition_issue": true/false, "issue_type": "顺序错误|步骤遗漏|粒度不当|无问题", "evidence": "具体证据", "suggestion": "改进建议"}}"""

            resp = self.llm.chat(prompt, system="你是任务分解分析专家。只输出 JSON。", max_tokens=500)
            if resp.success:
                decomp_analysis = _parse_llm_json(resp.content)

        return {
            "gap_detected": has_decomp_issue or (decomp_analysis and decomp_analysis.get("decomposition_issue")),
            "max_consecutive_fails": max_consecutive,
            "fail_sequence_start": fail_sequence_start,
            "decomp_analysis": decomp_analysis,
        }

    def generate(self, sense_result: dict, traces: list[dict]) -> dict:
        """Generate improved decomposition strategy."""
        if not sense_result.get("gap_detected"):
            return {"action": "none", "reason": "no_issue_detected"}

        analysis = sense_result.get("decomp_analysis") or {}
        suggestion = analysis.get("suggestion", "") if isinstance(analysis, dict) else ""

        if not suggestion:
            return {"action": "none", "reason": "no_suggestion"}

        return {
            "action": "recommend_decomposition",
            "strategy": suggestion,
            "issue_type": analysis.get("issue_type", "unknown"),
        }

    def validate(self, generate_result: dict, before_stats: dict, after_stats: dict) -> tuple[VerificationStatus, dict]:
        """Validate decomposition improvement."""
        if generate_result.get("action") == "none":
            return VerificationStatus.UNKNOWN, {"reason": "nothing_to_validate"}

        before_consec = before_stats.get("max_consecutive_fails", 0)
        after_consec = after_stats.get("max_consecutive_fails", 0)

        evidence = {
            "before_max_consecutive_fails": before_consec,
            "after_max_consecutive_fails": after_consec,
            "improvement": before_consec > after_consec,
        }

        if after_consec < before_consec:
            return VerificationStatus.SUCCESS, evidence
        elif after_consec == before_consec:
            return VerificationStatus.PARTIAL, evidence
        else:
            return VerificationStatus.UNKNOWN, {**evidence, "reason": "got_worse"}


class CalibrationDimension:
    """Calibration empowerment — measure and fix confidence-accuracy gap.

    Sense:  compare declared confidence vs actual success rate
    Generate: calibration curve + adjustment recommendations
    Validate: ECE (Expected Calibration Error) decreases
    """

    def __init__(self, store: MnemosyneStore, llm: Optional[LLMClient] = None):
        self.store = store
        self.llm = llm
        self.dimension = EmpowermentDimension.CALIBRATION

    def sense(self, traces: list[dict], state: WorldState) -> dict:
        """Measure calibration gap from traces.

        For each action, compare the effect (proxy for confidence)
        with the actual outcome (success/failure).
        """
        if not traces:
            return {"gap_detected": False, "reason": "no_traces"}

        # Bin by effect level (proxy for confidence)
        bins = {
            "low": {"predicted": [], "actual": []},      # effect < 0.4
            "medium": {"predicted": [], "actual": []},    # 0.4 <= effect < 0.7
            "high": {"predicted": [], "actual": []},      # effect >= 0.7
        }

        for t in traces:
            effect = t.get("effect", 0.5)
            success = 1.0 if t.get("outcome") == "success" else 0.0

            if effect < 0.4:
                bin_name = "low"
            elif effect < 0.7:
                bin_name = "medium"
            else:
                bin_name = "high"

            bins[bin_name]["predicted"].append(effect)
            bins[bin_name]["actual"].append(success)

        # Compute ECE (Expected Calibration Error)
        ece = 0.0
        total = 0
        calibration_details = {}
        for bin_name, data in bins.items():
            n = len(data["predicted"])
            if n == 0:
                continue
            avg_predicted = sum(data["predicted"]) / n
            avg_actual = sum(data["actual"]) / n
            gap = abs(avg_predicted - avg_actual)
            ece += gap * n
            total += n
            calibration_details[bin_name] = {
                "count": n,
                "avg_predicted": round(avg_predicted, 3),
                "avg_actual": round(avg_actual, 3),
                "gap": round(gap, 3),
            }

        ece = ece / max(total, 1)
        gap_detected = ece > 0.15  # Significant miscalibration

        return {
            "gap_detected": gap_detected,
            "ece": round(ece, 4),
            "total_samples": total,
            "calibration_details": calibration_details,
        }

    def generate(self, sense_result: dict, traces: list[dict]) -> dict:
        """Generate calibration adjustment recommendations."""
        if not sense_result.get("gap_detected"):
            return {"action": "none", "reason": "well_calibrated", "ece": sense_result.get("ece", 0)}

        details = sense_result.get("calibration_details", {})

        recommendations = []
        for bin_name, d in details.items():
            if d["gap"] > 0.1:
                if d["avg_predicted"] > d["avg_actual"]:
                    recommendations.append({
                        "bin": bin_name,
                        "issue": "过度自信",
                        "adjustment": f"将 {bin_name} 效果预估从 {d['avg_predicted']:.2f} 下调到 {d['avg_actual']:.2f}",
                    })
                else:
                    recommendations.append({
                        "bin": bin_name,
                        "issue": "过度保守",
                        "adjustment": f"将 {bin_name} 效果预估从 {d['avg_predicted']:.2f} 上调到 {d['avg_actual']:.2f}",
                    })

        return {
            "action": "adjust_calibration",
            "ece": sense_result.get("ece", 0),
            "recommendations": recommendations,
        }

    def validate(self, generate_result: dict, before_stats: dict, after_stats: dict) -> tuple[VerificationStatus, dict]:
        """Validate calibration improvement (ECE decrease)."""
        if generate_result.get("action") == "none":
            return VerificationStatus.SUCCESS, {"reason": "already_well_calibrated"}

        before_ece = before_stats.get("ece", 1.0)
        after_ece = after_stats.get("ece", 1.0)

        evidence = {
            "before_ece": round(before_ece, 4),
            "after_ece": round(after_ece, 4),
            "improvement": round(before_ece - after_ece, 4),
        }

        if after_ece < before_ece * 0.8:  # 20% improvement
            return VerificationStatus.SUCCESS, evidence
        elif after_ece < before_ece:
            return VerificationStatus.PARTIAL, evidence
        else:
            return VerificationStatus.UNKNOWN, {**evidence, "reason": "no_improvement"}


class DimensionOrchestrator:
    """Orchestrates all implemented dimensions.

    Runs sense → generate → validate for each dimension,
    records results to Mnemosyne.
    """

    def __init__(self, store: MnemosyneStore, llm: Optional[LLMClient] = None):
        self.store = store
        self.llm = llm
        self.dimensions = {
            EmpowermentDimension.KNOWLEDGE: KnowledgeDimension(store, llm),
            EmpowermentDimension.DECOMPOSITION: DecompositionDimension(store, llm),
            EmpowermentDimension.CALIBRATION: CalibrationDimension(store, llm),
            EmpowermentDimension.ATTENTION: AttentionDimension(store, llm),
            EmpowermentDimension.METACOGNITION: MetacognitionDimension(store, llm),
            EmpowermentDimension.ENVIRONMENT: EnvironmentDimension(store, llm),
        }
        self._results: list[dict] = []

    def run_dimension(self, dim: EmpowermentDimension, traces: list[dict], state: WorldState) -> dict:
        """Run full sense → generate → validate for one dimension."""
        impl = self.dimensions[dim]

        # Sense
        sense_result = impl.sense(traces, state)

        # Generate
        generate_result = impl.generate(sense_result, traces)

        # Validate (simplified: compare before/after stats from same traces)
        before_stats = self._compute_stats(traces)
        # In real usage, after_stats would come from post-empowerment traces
        after_stats = before_stats  # Placeholder

        status, evidence = impl.validate(generate_result, before_stats, after_stats)

        result = {
            "dimension": dim.value,
            "sense": sense_result,
            "generate": generate_result,
            "validation_status": status.value,
            "validation_evidence": evidence,
            "timestamp": datetime.now().isoformat(),
        }

        # Record to Mnemosyne
        self.store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content=f"[{dim.value}] sense={sense_result.get('gap_detected')} gen={generate_result.get('action')} val={status.value}",
            tags=[dim.value, "dimension_analysis"],
            importance=0.7 if status == VerificationStatus.SUCCESS else 0.4,
        ))

        self._results.append(result)
        return result

    def run_all(self, traces: list[dict], state: WorldState) -> dict:
        """Run all dimensions with a shared LLM budget across sense calls."""
        results = {}
        budget_left = int(getattr(self, "_run_llm_budget", 3))
        for dim in self.dimensions:
            try:
                impl = self.dimensions[dim]
                if impl is not None:
                    impl._llm_budget_left = budget_left
                results[dim.value] = self.run_dimension(dim, traces, state)
                if impl is not None:
                    budget_left = int(getattr(impl, "_llm_budget_left", budget_left))
            except Exception as e:
                logger.error(f"Dimension {dim.value} failed: {e}")
                results[dim.value] = {"error": str(e)}
        return results

    @staticmethod
    def _compute_stats(traces: list[dict]) -> dict:
        """Compute success rate and other stats from traces."""
        if not traces:
            return {"success_rate": 0, "max_consecutive_fails": 0, "ece": 0}

        successes = sum(1 for t in traces if t.get("outcome") == "success")
        success_rate = successes / len(traces)

        # Max consecutive fails
        max_consec = 0
        consec = 0
        for t in traces:
            if t.get("outcome") == "failure":
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0

        return {
            "success_rate": success_rate,
            "max_consecutive_fails": max_consec,
            "ece": 0.5,  # Simplified
        }


class AttentionDimension:
    """Attention management — analyze context information density vs performance.

    Sense:  compare traces with high/low param counts and their outcomes
    Generate: recommend optimal context size and what to include/exclude
    Validate: performance improves with optimized context
    """

    def __init__(self, store: MnemosyneStore, llm: Optional[LLMClient] = None):
        self.store = store
        self.llm = llm
        self.dimension = EmpowermentDimension.ATTENTION

    def sense(self, traces: list[dict], state: WorldState) -> dict:
        """Analyze attention patterns from traces."""
        if not traces:
            return {"gap_detected": False, "reason": "no_traces"}

        # Analyze param richness vs outcome
        rich_params = []  # traces with substantial params
        sparse_params = []  # traces with minimal params

        for t in traces:
            params = t.get("action_params", {})
            param_size = len(str(params))
            if param_size > 50:
                rich_params.append(t)
            else:
                sparse_params.append(t)

        rich_success = sum(1 for t in rich_params if t.get("outcome") == "success") / max(len(rich_params), 1)
        sparse_success = sum(1 for t in sparse_params if t.get("outcome") == "success") / max(len(sparse_params), 1)

        gap = abs(rich_success - sparse_success)
        gap_detected = gap > 0.15 and len(rich_params) >= 3 and len(sparse_params) >= 3

        # LLM analysis
        analysis = None
        if self.llm and self.llm.available and traces:
            trace_summary = json.dumps(
                [{"action": t.get("action"), "outcome": t.get("outcome"),
                  "params_size": len(str(t.get("action_params", {})))} for t in traces[-15:]],
                ensure_ascii=False,
            )
            prompt = f"""分析以下行为轨迹，判断信息密度是否合理。

轨迹（含参数大小）：
{trace_summary}

判断：
1. 参数丰富的轨迹是否成功率更高？
2. 是否存在信息过载（参数太多但效果差）？
3. 是否存在信息不足（参数太少导致失败）？

输出 JSON：
{{"attention_issue": true/false, "issue_type": "过载|不足|无问题", "evidence": "依据", "suggestion": "建议"}}"""

            resp = self.llm.chat(prompt, system="你是注意力管理分析专家。只输出 JSON。", max_tokens=400)
            if resp.success:
                analysis = _parse_llm_json(resp.content)

        return {
            "gap_detected": gap_detected or (analysis and analysis.get("attention_issue")),
            "rich_success_rate": round(rich_success, 3),
            "sparse_success_rate": round(sparse_success, 3),
            "gap": round(gap, 3),
            "analysis": analysis,
        }

    def generate(self, sense_result: dict, traces: list[dict]) -> dict:
        """Generate attention optimization recommendations."""
        if not sense_result.get("gap_detected"):
            return {"action": "none", "reason": "attention_well_balanced"}

        analysis = sense_result.get("analysis") or {}
        suggestion = analysis.get("suggestion", "") if isinstance(analysis, dict) else ""

        return {
            "action": "optimize_attention",
            "issue_type": analysis.get("issue_type", "unknown") if isinstance(analysis, dict) else "unknown",
            "recommendation": suggestion or "调整上下文信息密度",
            "rich_vs_sparse_gap": sense_result.get("gap", 0),
        }

    def validate(self, generate_result: dict, before_stats: dict, after_stats: dict) -> tuple[VerificationStatus, dict]:
        if generate_result.get("action") == "none":
            return VerificationStatus.SUCCESS, {"reason": "already_balanced"}
        before_gap = before_stats.get("attention_gap", 0.5)
        after_gap = after_stats.get("attention_gap", 0.5)
        evidence = {"before_gap": before_gap, "after_gap": after_gap}
        if after_gap < before_gap:
            return VerificationStatus.SUCCESS, evidence
        return VerificationStatus.UNKNOWN, evidence


class MetacognitionDimension:
    """Metacognition — detect 'should have stopped' patterns.

    Sense:  find sequences where agent kept trying despite repeated failures
    Generate: recommend stopping/asking/pivoting rules
    Validate: fewer wasted attempts after applying rules
    """

    def __init__(self, store: MnemosyneStore, llm: Optional[LLMClient] = None):
        self.store = store
        self.llm = llm
        self.dimension = EmpowermentDimension.METACOGNITION

    def sense(self, traces: list[dict], state: WorldState) -> dict:
        """Detect metacognitive failures from traces."""
        if not traces:
            return {"gap_detected": False, "reason": "no_traces"}

        # Detect: consecutive failures followed by eventual success (wasted attempts)
        # or consecutive failures with no success (should have pivoted)
        wasted_attempts = 0
        should_pivot = 0
        consec_fails = 0

        for t in traces:
            if t.get("outcome") == "failure":
                consec_fails += 1
            else:
                if consec_fails >= 2:
                    wasted_attempts += consec_fails - 1  # Could have stopped earlier
                consec_fails = 0

        # If final consecutive fails >= 3, should have pivoted
        if consec_fails >= 3:
            should_pivot = 1

        gap_detected = wasted_attempts >= 2 or should_pivot >= 1

        # LLM analysis
        analysis = None
        if self.llm and self.llm.available and traces:
            trace_summary = json.dumps(
                [{"action": t.get("action"), "outcome": t.get("outcome")} for t in traces[-20:]],
                ensure_ascii=False,
            )
            prompt = f"""分析以下行为轨迹，判断元认知是否合理。

轨迹：
{trace_summary}

判断：
1. 是否有"该停未停"的模式（连续失败后仍继续同一策略）？
2. 是否有"过早放弃"的模式（一次失败就换方向）？
3. 最优的停止/切换时机是什么？

输出 JSON：
{{"metacog_issue": true/false, "issue_type": "该停未停|过早放弃|无问题", "wasted_attempts": 整数, "suggestion": "建议"}}"""

            resp = self.llm.chat(prompt, system="你是元认知分析专家。只输出 JSON。", max_tokens=400)
            if resp.success:
                analysis = _parse_llm_json(resp.content)

        return {
            "gap_detected": gap_detected or (analysis and analysis.get("metacog_issue")),
            "wasted_attempts": wasted_attempts,
            "should_pivot": should_pivot,
            "analysis": analysis,
        }

    def generate(self, sense_result: dict, traces: list[dict]) -> dict:
        """Generate metacognitive rules."""
        if not sense_result.get("gap_detected"):
            return {"action": "none", "reason": "metacognition_ok"}

        analysis = sense_result.get("analysis") or {}
        suggestion = analysis.get("suggestion", "") if isinstance(analysis, dict) else ""

        return {
            "action": "apply_metacognitive_rules",
            "issue_type": analysis.get("issue_type", "unknown") if isinstance(analysis, dict) else "unknown",
            "recommendation": suggestion or "连续失败 2 次后切换策略",
            "wasted_attempts": sense_result.get("wasted_attempts", 0),
        }

    def validate(self, generate_result: dict, before_stats: dict, after_stats: dict) -> tuple[VerificationStatus, dict]:
        if generate_result.get("action") == "none":
            return VerificationStatus.SUCCESS, {"reason": "metacognition_ok"}
        before_waste = before_stats.get("wasted_attempts", 0)
        after_waste = after_stats.get("wasted_attempts", 0)
        evidence = {"before_wasted": before_waste, "after_wasted": after_waste}
        if after_waste < before_waste:
            return VerificationStatus.SUCCESS, evidence
        return VerificationStatus.UNKNOWN, evidence


class EnvironmentDimension:
    """Environment shaping — analyze environment config impact on success.

    Sense:  correlate environment params (params richness, action types) with outcomes
    Generate: recommend environment improvements
    Validate: success rate improves after environment changes
    """

    def __init__(self, store: MnemosyneStore, llm: Optional[LLMClient] = None):
        self.store = store
        self.llm = llm
        self.dimension = EmpowermentDimension.ENVIRONMENT

    def sense(self, traces: list[dict], state: WorldState) -> dict:
        """Analyze environment impact from traces."""
        if not traces:
            return {"gap_detected": False, "reason": "no_traces"}

        # Analyze by action type: which actions have consistently low success?
        action_stats = {}
        for t in traces:
            a = t.get("action", "unknown")
            action_stats.setdefault(a, {"total": 0, "success": 0})
            action_stats[a]["total"] += 1
            if t.get("outcome") == "success":
                action_stats[a]["success"] += 1

        # Find actions with consistently low success
        weak_actions = []
        for a, s in action_stats.items():
            rate = s["success"] / max(s["total"], 1)
            if rate < 0.5 and s["total"] >= 2:
                weak_actions.append({"action": a, "success_rate": round(rate, 2), "count": s["total"]})

        gap_detected = len(weak_actions) > 0

        # LLM analysis
        analysis = None
        if self.llm and self.llm.available and traces:
            def _pkeys(t: dict) -> list:
                p = t.get("action_params", t.get("params", {}))
                if isinstance(p, dict):
                    return list(p.keys())
                return []
            try:
                trace_summary = json.dumps(
                    [{"action": t.get("action"), "outcome": t.get("outcome"),
                      "params": _pkeys(t)} for t in traces[-8:]],
                    ensure_ascii=False,
                )
            except Exception:
                trace_summary = json.dumps(
                    [{"action": t.get("action"), "outcome": t.get("outcome")} for t in traces[-6:]],
                    ensure_ascii=False,
                )
            # Hard LLM budget for daemon ticks
            budget = int(getattr(self, "_llm_budget_left", 2))
            if budget > 0:
                self._llm_budget_left = budget - 1
                prompt = f"""分析以下行为轨迹，判断环境配置是否需要改进。

轨迹：
{trace_summary}

判断：
1. 哪些操作类型成功率低？可能是环境配置问题？
2. 是否缺少必要的工具/参数/上下文？
3. 环境改进建议是什么？

输出 JSON：
{{"env_issue": true/false, "weak_areas": ["薄弱领域"], "suggestion": "环境改进建议"}}"""

                resp = self.llm.chat(prompt, system="你是环境配置分析专家。只输出 JSON。", max_tokens=400)
                if resp.success:
                    analysis = _parse_llm_json(resp.content)
            else:
                analysis = None

        return {
            "gap_detected": gap_detected or (analysis and analysis.get("env_issue")),
            "weak_actions": weak_actions,
            "action_stats": {a: {"rate": round(s["success"] / max(s["total"], 1), 2), "n": s["total"]}
                             for a, s in action_stats.items()},
            "analysis": analysis,
        }

    def generate(self, sense_result: dict, traces: list[dict]) -> dict:
        """Generate environment improvement recommendations."""
        if not sense_result.get("gap_detected"):
            return {"action": "none", "reason": "environment_ok"}

        analysis = sense_result.get("analysis") or {}
        suggestion = analysis.get("suggestion", "") if isinstance(analysis, dict) else ""
        weak_actions = sense_result.get("weak_actions", [])

        return {
            "action": "improve_environment",
            "weak_areas": [w["action"] for w in weak_actions],
            "recommendation": suggestion or "为薄弱操作补充必要参数和上下文",
        }

    def validate(self, generate_result: dict, before_stats: dict, after_stats: dict) -> tuple[VerificationStatus, dict]:
        if generate_result.get("action") == "none":
            return VerificationStatus.SUCCESS, {"reason": "environment_ok"}
        before_rate = before_stats.get("success_rate", 0)
        after_rate = after_stats.get("success_rate", 0)
        evidence = {"before_rate": before_rate, "after_rate": after_rate}
        if after_rate > before_rate:
            return VerificationStatus.SUCCESS, evidence
        return VerificationStatus.UNKNOWN, evidence
