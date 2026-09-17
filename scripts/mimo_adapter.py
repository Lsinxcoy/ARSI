"""MiMo Desktop Adapter — export agent behavior traces to ARSI.

Extracts behavior from a MiMo Desktop conversation session
and ingests it into ARSI for analysis and empowerment.

Usage:
    python scripts/mimo_adapter.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.core import ARSI


# Behavior traces extracted from this conversation session
# Each trace = one meaningful action + outcome
CONVERSATION_TRACES = [
    # Research phase
    {"action": "webfetch_github_repos", "outcome": "success", "effect": 0.9,
     "params": {"target": "Lsinxcoy profile", "found": "38 repos"}},
    {"action": "webfetch_paper_metarsi", "outcome": "success", "effect": 0.9,
     "params": {"target": "arXiv:2609.06396", "pages": "47"}},
    {"action": "webfetch_paper_mwm", "outcome": "success", "effect": 0.9,
     "params": {"target": "arXiv:2607.27201"}},
    {"action": "analyze_repos", "outcome": "success", "effect": 0.8,
     "params": {"repos_analyzed": 38, "timeline": "2026-05 to 2026-09"}},

    # Architecture design phase
    {"action": "design_architecture", "outcome": "success", "effect": 0.85,
     "params": {"output": "ARSI whitepaper v0.8", "sections": 12}},
    {"action": "design_siwm", "outcome": "success", "effect": 0.85,
     "params": {"output": "SIWM technical spec", "layers": 3}},
    {"action": "design_memory", "outcome": "success", "effect": 0.8,
     "params": {"output": "Mnemosyne three-zone design"}},
    {"action": "design_empowerment", "outcome": "success", "effect": 0.8,
     "params": {"output": "Nine empowerment dimensions"}},
    {"action": "design_implementation", "outcome": "success", "effect": 0.85,
     "params": {"output": "Code-level blueprint", "phases": 5}},

    # Implementation phase
    {"action": "create_project_structure", "outcome": "success", "effect": 0.9,
     "params": {"dirs": 13, "files": 29}},
    {"action": "write_foundation_layer", "outcome": "success", "effect": 0.9,
     "params": {"files": ["schema.py", "store.py", "iron_laws.py", "config.py"]}},
    {"action": "write_mnemosyne", "outcome": "success", "effect": 0.85,
     "params": {"files": ["core.py", "memory_proxy.py"]}},
    {"action": "write_siwm", "outcome": "success", "effect": 0.85,
     "params": {"files": ["siwm.py"], "components": ["eta", "mindzero", "layer1"]}},
    {"action": "write_governor", "outcome": "success", "effect": 0.85,
     "params": {"files": ["core.py"], "components": ["decision_loop", "dim_manager"]}},
    {"action": "write_empowerment", "outcome": "success", "effect": 0.8,
     "params": {"files": ["engine.py"]}},
    {"action": "write_dream_pipeline", "outcome": "success", "effect": 0.8,
     "params": {"files": ["dream.py"]}},
    {"action": "write_sealed_eval", "outcome": "success", "effect": 0.75,
     "params": {"files": ["evaluator.py"], "tasks": 12}},

    # Testing phase
    {"action": "run_tests", "outcome": "success", "effect": 0.9,
     "params": {"total": 69, "passed": 69, "failed": 0}},
    {"action": "fix_test_failures", "outcome": "success", "effect": 0.7,
     "params": {"fixes": 4, "issues": ["yaml_encoding", "pydantic_attr", "dim_data", "llm_network"]}},

    # LLM integration phase
    {"action": "install_dependencies", "outcome": "success", "effect": 0.8,
     "params": {"packages": ["openai", "httpx", "pydantic", "pyyaml"]}},
    {"action": "test_llm_opencode", "outcome": "failure", "effect": 0.0,
     "params": {"model": "union-alpha", "error": "500_internal_server"}},
    {"action": "test_llm_opencode_omen", "outcome": "success", "effect": 0.8,
     "params": {"model": "omen-alpha", "provider": "opencode.ai"}},
    {"action": "test_llm_openrouter", "outcome": "success", "effect": 0.9,
     "params": {"model": "stealth/union-alpha", "provider": "openrouter"}},
    {"action": "integrate_llm_brain", "outcome": "success", "effect": 0.85,
     "params": {"modules": ["governor", "mindzero", "diagnosis", "dream"]}},

    # Core + scripts phase
    {"action": "write_arsi_core", "outcome": "success", "effect": 0.9,
     "params": {"file": "core.py", "lines": 387}},
    {"action": "write_cold_start", "outcome": "success", "effect": 0.8,
     "params": {"file": "cold_start.py", "phases": 7}},
    {"action": "write_human_cli", "outcome": "success", "effect": 0.8,
     "params": {"file": "human_cli.py", "commands": 10}},

    # Run phase
    {"action": "run_cold_start", "outcome": "success", "effect": 0.85,
     "params": {"warmup": 30, "llm_calls": 3, "dreams": 3, "eta": 0.25}},
    {"action": "run_term_with_llm", "outcome": "success", "effect": 0.8,
     "params": {"steps": 5, "llm_decisions": 2, "heuristic_fallbacks": 3}},

    # Git phase
    {"action": "git_push", "outcome": "success", "effect": 0.9,
     "params": {"commits": 6, "repo": "Lsinxcoy/ARSI"}},

    # Failures / issues encountered
    {"action": "yaml_encoding_error", "outcome": "failure", "effect": 0.0,
     "params": {"file": "iron_laws.yaml", "fix": "rewrote_as_plain_yaml"}},
    {"action": "ps_escape_error", "outcome": "failure", "effect": 0.0,
     "params": {"cause": "PowerShell backtick escaping", "fix": "use write tool"}},
    {"action": "pydantic_attr_error", "outcome": "failure", "effect": 0.0,
     "params": {"cause": "non-annotated class attr", "fix": "use plain class"}},
    {"action": "llm_timeout_in_tests", "outcome": "failure", "effect": 0.0,
     "params": {"cause": "tests making real network calls", "fix": "disable llm in fixture"}},
    {"action": "union_alpha_500", "outcome": "failure", "effect": 0.0,
     "params": {"cause": "opencode.ai server-side error", "fix": "switch to openrouter"}},
]


def main():
    print("=" * 60)
    print("MiMo Desktop → ARSI 行为轨迹导入")
    print("=" * 60)

    # Build ARSI
    arsi = ARSI.from_config("config/arsi.yaml")

    print(f"\n导入 {len(CONVERSATION_TRACES)} 条行为轨迹...")

    # Ingest traces
    for i, trace in enumerate(CONVERSATION_TRACES):
        arsi.ingest_trace(
            agent_id="mimo-desktop",
            action=trace["action"],
            outcome=trace["outcome"],
            effect=trace["effect"],
            params=trace["params"],
        )

    print(f"导入完成。Store stats:")
    stats = arsi.store.get_stats()
    print(f"  轨迹: {stats['trace_count']}")
    print(f"  经验: {stats['experience_count']}")
    print(f"  代理: {stats['proxy_count']}")

    # Train predictor
    print(f"\n训练行为预测器...")
    train_result = arsi.siwm.train_from_history()
    print(f"  规则数: {train_result.get('rule_count', 0)}")

    # Run LLM-powered analysis
    print(f"\n运行 LLM 分析...")

    # 1. MindZero inference
    traces = arsi.store.get_recent_traces(n=30)
    mental = arsi.llm_mindzero.infer_mental_state(traces)
    print(f"\n── 心智状态推断 ──")
    print(f"  信念 ({len(mental.beliefs)}):")
    for b in mental.beliefs[:5]:
        print(f"    - {b.content} (conf={b.confidence:.2f})")
    print(f"  情绪: {mental.affect}")

    # 2. Governor decision
    state = arsi.siwm.refresh_state()
    candidates = ["learn", "evolve", "dream", "maintain"]
    decision = arsi.llm_governor.decide(state, candidates)
    print(f"\n── Governor 决策 ──")
    print(f"  动作: {decision['action']}")
    print(f"  理由: {decision['reason']}")
    print(f"  风险: {decision['risk']}")

    # 3. Empowerment diagnosis
    diagnosis = arsi.llm_diagnosis.diagnose(traces, state)
    print(f"\n── 赋能诊断 ──")
    print(f"  薄弱维度: {diagnosis.get('dimension', '?')}")
    print(f"  诊断理由: {diagnosis.get('reason', '?')}")
    print(f"  改进建议: {diagnosis.get('recommendation', '?')}")
    print(f"  置信度: {diagnosis.get('confidence', '?')}")

    # 4. Run a few steps
    print(f"\n运行 3 步...")
    for i in range(3):
        result = arsi.step()
        d = result["decision"]
        print(f"  Step {result['step']}: {d['action']} [{d.get('source', '?')}]")

    # Final stats
    print(f"\n── 最终状态 ──")
    final = arsi.get_stats()
    for k in ["trace_count", "experience_count", "eta", "step_count",
              "llm_governor_calls", "llm_mindzero_calls"]:
        print(f"  {k}: {final.get(k, '?')}")

    arsi.close()
    print(f"\n完成。")


if __name__ == "__main__":
    main()
