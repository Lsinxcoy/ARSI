"""Phase A Demo — three empowerment dimensions on MiMo conversation traces.

Usage:
    set ARSI_API_KEY=...
    python scripts/demo_dimensions.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.core import ARSI
from arsi.empowerment.dimensions import DimensionOrchestrator
from arsi.foundation.llm import LLMClient, LLMConfig


def main():
    print("=" * 60)
    print("Phase A: 三个赋能维度深度分析")
    print("=" * 60)

    # Build ARSI and ingest MiMo traces
    arsi = ARSI.from_config("config/arsi.yaml")

    print("\n[1/4] 导入 MiMo 对话轨迹...")
    from scripts.mimo_adapter import CONVERSATION_TRACES
    for t in CONVERSATION_TRACES:
        arsi.ingest_trace(
            agent_id="mimo-desktop",
            action=t["action"],
            outcome=t["outcome"],
            effect=t["effect"],
            params=t["params"],
        )
    print(f"  导入 {len(CONVERSATION_TRACES)} 条轨迹")

    # Get traces and state
    traces = arsi.store.get_recent_traces(n=100)
    state = arsi.siwm.refresh_state()

    print(f"  总轨迹: {len(traces)}")
    print(f"  η: {state.eta:.4f}")

    # Run dimension analysis
    print("\n[2/4] 运行三维度分析...")
    orch = DimensionOrchestrator(arsi.store, llm=arsi.llm)
    results = orch.run_all(traces, state)

    for dim_name, result in results.items():
        print(f"\n── {dim_name.upper()} ──")
        if "error" in result:
            print(f"  错误: {result['error']}")
            continue

        sense = result.get("sense", {})
        gen = result.get("generate", {})
        val_status = result.get("validation_status", "?")
        val_evidence = result.get("validation_evidence", {})

        print(f"  感知: gap_detected={sense.get('gap_detected')}")
        if dim_name == "knowledge":
            print(f"    失败数: {sense.get('failure_count', 0)}")
            print(f"    高失败动作: {sense.get('high_fail_actions', [])}")
            kg = sense.get("knowledge_gap")
            if kg:
                print(f"    知识缺口: {kg.get('missing_knowledge', '?')}")
        elif dim_name == "decomposition":
            print(f"    最大连续失败: {sense.get('max_consecutive_fails', 0)}")
            da = sense.get("decomp_analysis")
            if da:
                print(f"    分解问题: {da.get('issue_type', '?')}")
                print(f"    建议: {da.get('suggestion', '?')}")
        elif dim_name == "calibration":
            print(f"    ECE: {sense.get('ece', '?')}")
            details = sense.get("calibration_details", {})
            for b, d in details.items():
                print(f"    {b}: 预测={d.get('avg_predicted')} 实际={d.get('avg_actual')} 差距={d.get('gap')}")

        print(f"  生成: {gen.get('action', '?')}")
        if gen.get("recommendations"):
            for r in gen["recommendations"]:
                print(f"    - {r.get('issue')}: {r.get('adjustment')}")
        if gen.get("items"):
            print(f"    知识条目: {gen.get('count', 0)} 条")

        print(f"  验证: {val_status}")
        if val_evidence:
            print(f"    证据: {json.dumps(val_evidence, ensure_ascii=False)[:100]}")

    # Summary
    print("\n[3/4] 综合评估...")
    total_gaps = sum(1 for r in results.values() if r.get("sense", {}).get("gap_detected"))
    print(f"  检测到缺口的维度: {total_gaps}/3")

    print("\n[4/4] 最终状态...")
    stats = arsi.get_stats()
    print(f"  轨迹: {stats['trace_count']}")
    print(f"  经验: {stats['experience_count']}")
    print(f"  η: {stats['eta']}")

    arsi.close()
    print("\n完成。")


if __name__ == "__main__":
    # Add parent to path for mimo_adapter import
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
