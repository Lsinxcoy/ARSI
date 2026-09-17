"""ARSI LLM Integration Demo — shows the system working with a real LLM.

Usage:
    set ARSI_API_KEY=sk-...
    python scripts/demo_llm.py
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.foundation.llm import LLMClient, LLMConfig
from arsi.foundation.schema import BehaviorTrace, WorldState
from arsi.foundation.store import MnemosyneStore
from arsi.foundation.iron_laws import IronLaws
from arsi.mnemosyne.core import Mnemosyne
from arsi.world_model.siwm import SIWM, MindZero
from arsi.governor.core import AutopoieticGovernor, DimensionManager
from arsi.empowerment.engine import EmpowermentEngine
from arsi.pipelines.dream import DreamPipeline


def create_llm() -> LLMClient:
    """Create LLM client with proxy support."""
    api_key = os.environ.get("ARSI_API_KEY", "")
    config = LLMConfig(
        provider="openai",
        model="omen-alpha",
        api_key=api_key,
        api_base="https://opencode.ai/zen/go/v1",
        use_proxy=True,
        proxy_url="http://127.0.0.1:7890",
        extra_headers={"x-opencode-session": "arsi-demo-001"},
        temperature=0.3,
        max_tokens=1024,
        fallback_to_heuristic=True,
    )
    return LLMClient(config)


def llm_mindzero(llm: LLMClient, traces: list[dict]) -> dict:
    """Use LLM to infer mental state from behavior traces."""
    trace_summary = json.dumps(traces[-10:], ensure_ascii=False, default=str)
    prompt = f"""你是一个 AI 系统的心智状态推断器。分析以下行为轨迹，推断系统的心智状态。

行为轨迹（最近10条）：
{trace_summary}

请以 JSON 格式输出：
{{
  "beliefs": ["系统可能相信什么"],
  "goals": ["系统当前可能在追求什么"],
  "affect": {{"管道名": 情绪效价(-1到1)}},
  "confidence": 0.0到1.0的置信度
}}"""

    response = llm.chat(prompt, system="你是一个精确的心智状态分析器。只输出 JSON。")
    if response.success:
        try:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {"error": "parse_failed", "raw": response.content[:200]}
    return {"error": response.error}


def llm_governor_decision(llm: LLMClient, state: WorldState, candidates: list[str]) -> dict:
    """Use LLM to help Governor decide."""
    prompt = f"""你是一个自创生 AI 系统的 Governor。根据当前系统状态，选择下一步动作。

系统状态：
- 代数: {state.phi.generation}
- η（自我模型失配度）: {state.eta:.3f}
- 存储: {state.phi.storage_stats}
- 信念数: {len(state.psi.beliefs)}
- 管道健康: {state.phi.pipeline_health}

候选动作: {candidates}

请以 JSON 格式输出决策：
{{
  "action": "选择的动作",
  "reason": "为什么选这个",
  "expected_effect": "预期效果",
  "risk": "low|medium|high"
}}"""

    response = llm.chat(prompt, system="你是一个理性的系统调度器。只输出 JSON。")
    if response.success:
        try:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {"action": "remember", "reason": "llm_parse_failed", "risk": "low"}
    return {"action": "remember", "reason": "llm_unavailable", "risk": "low"}


def llm_diagnose_weakness(llm: LLMClient, traces: list[dict]) -> dict:
    """Use LLM to diagnose agent weakness."""
    trace_summary = json.dumps(traces[-20:], ensure_ascii=False, default=str)
    prompt = f"""你是一个 AI agent 的诊断专家。分析以下行为轨迹，找出最需要改进的维度。

行为轨迹（最近20条）：
{trace_summary}

九个赋能维度：
- skill: 技能注入
- harness: 脚手架配置
- knowledge: 知识注入
- decomposition: 任务分解策略
- metacognition: 元认知（何时停/问/放弃）
- attention: 注意力管理
- environment: 环境塑形
- social_graph: 社交图谱
- calibration: 置信度校准

请以 JSON 格式输出：
{{
  "dimension": "最需要改进的维度",
  "reason": "诊断依据",
  "recommendation": "具体改进建议",
  "confidence": 0.0到1.0
}}"""

    response = llm.chat(prompt, system="你是一个精确的诊断专家。只输出 JSON。")
    if response.success:
        try:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {"dimension": "knowledge", "reason": "llm_parse_failed"}
    return {"dimension": "knowledge", "reason": "llm_unavailable"}


def main():
    print("=" * 60)
    print("ARSI LLM Integration Demo")
    print("=" * 60)

    # 1. Setup
    print("\n[1/6] 初始化组件...")
    llm = create_llm()
    print(f"  LLM available: {llm.available}")
    print(f"  Model: {llm.config.model}")
    print(f"  Proxy: {llm.config.proxy_url}")

    store = MnemosyneStore(":memory:")
    mnemosyne = Mnemosyne(store)
    siwm = SIWM(store)
    laws = IronLaws("config/iron_laws.yaml")
    governor = AutopoieticGovernor(siwm, store, laws, DimensionManager())

    # 2. Test LLM connectivity
    print("\n[2/6] 测试 LLM 连通性...")
    resp = llm.chat("Reply with exactly: ARSI_ONLINE", max_tokens=20)
    print(f"  Response: {resp.content}")
    print(f"  Success: {resp.success}")

    # 3. Ingest behavior traces
    print("\n[3/6] 模拟行为轨迹摄入...")
    for i in range(15):
        trace = BehaviorTrace(
            agent_id="agent_alpha",
            action=["learn", "evolve", "reflect", "remember"][i % 4],
            outcome="success" if i % 3 != 0 else "failure",
            effect=0.3 + (i * 0.04),
        )
        mnemosyne.ingest_trace(trace)
    print(f"  Ingested 15 traces")
    print(f"  Store stats: {mnemosyne.get_stats()}")

    # 4. LLM-powered MindZero
    print("\n[4/6] LLM 心智状态推断 (MindZero)...")
    traces = store.get_recent_traces(n=10)
    mental = llm_mindzero(llm, traces)
    print(f"  Beliefs: {mental.get('beliefs', ['N/A'])}")
    print(f"  Goals: {mental.get('goals', ['N/A'])}")
    print(f"  Affect: {mental.get('affect', {})}")
    print(f"  Confidence: {mental.get('confidence', 'N/A')}")

    # 5. LLM-powered Governor decision
    print("\n[5/6] LLM Governor 决策...")
    state = siwm.refresh_state()
    candidates = ["dream", "learn", "evolve", "maintain", "remember"]
    decision = llm_governor_decision(llm, state, candidates)
    print(f"  Action: {decision.get('action')}")
    print(f"  Reason: {decision.get('reason')}")
    print(f"  Expected: {decision.get('expected_effect')}")
    print(f"  Risk: {decision.get('risk')}")

    # 6. LLM-powered diagnosis
    print("\n[6/6] LLM 赋能诊断...")
    diagnosis = llm_diagnose_weakness(llm, traces)
    print(f"  Dimension: {diagnosis.get('dimension')}")
    print(f"  Reason: {diagnosis.get('reason')}")
    print(f"  Recommendation: {diagnosis.get('recommendation')}")
    print(f"  Confidence: {diagnosis.get('confidence')}")

    # Summary
    print("\n" + "=" * 60)
    print("Demo 完成！LLM 已成功接入 ARSI 核心模块。")
    print("=" * 60)
    print("\n接入点：")
    print("  1. MindZero — 用 LLM 从行为轨迹推断心智状态")
    print("  2. Governor — 用 LLM 辅助决策下一步动作")
    print("  3. 赋能诊断 — 用 LLM 分析 agent 薄弱环节")
    print("  4. 梦境管道 — 可用 LLM 做 belief 调和（待接入）")


if __name__ == "__main__":
    main()
