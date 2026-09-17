"""ARSI Cold Start — bootstrap the system from zero.

Usage:
    set ARSI_API_KEY=sk-...
    python scripts/cold_start.py [--warmup 50] [--agent agent_alpha]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.core import ARSI
from arsi.foundation.schema import MemoryZone


def main():
    parser = argparse.ArgumentParser(description="ARSI cold start bootstrap")
    parser.add_argument("--config", default="config/arsi.yaml")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup cycles")
    parser.add_argument("--agent", default="agent_alpha", help="Primary agent ID")
    args = parser.parse_args()

    print("=" * 60)
    print("ARSI Cold Start")
    print("=" * 60)

    # Phase 1: Initialize
    print("\n[Phase 1] 初始化系统...")
    arsi = ARSI.from_config(args.config)
    llm_status = "available" if arsi.llm and arsi.llm.available else "unavailable"
    print(f"  LLM: {llm_status}")
    print(f"  Iron laws: {len(arsi.iron_laws.law_ids)} loaded")
    print(f"  Store: {arsi.store._db_path}")

    # Phase 2: Baseline stats
    print("\n[Phase 2] 基线状态...")
    stats = arsi.get_stats()
    print(f"  Generation: {stats['generation']}")
    print(f"  Traces: {stats['trace_count']}")
    print(f"  Experiences: {stats['experience_count']}")
    print(f"  η: {stats['eta']}")

    # Phase 3: Warmup with simulated traces
    print(f"\n[Phase 3] 预热 {args.warmup} 个周期...")
    actions = ["learn", "evolve", "reflect", "remember", "maintain"]
    outcomes = ["success", "failure", "success", "success", "partial"]

    for i in range(args.warmup):
        action = actions[i % len(actions)]
        outcome = outcomes[i % len(outcomes)]
        effect = 0.3 + (i % 10) * 0.07

        arsi.ingest_trace(
            agent_id=args.agent,
            action=action,
            outcome=outcome,
            effect=effect,
            params={"warmup_cycle": i, "source": "cold_start"},
        )

        if (i + 1) % 10 == 0:
            print(f"  周期 {i+1}/{args.warmup} (η={arsi.siwm.eta.eta_smooth:.3f})")

    print(f"  预热完成。轨迹: {arsi.store.get_stats()['trace_count']}")

    # Phase 4: Train Layer 1 predictor
    print("\n[Phase 4] 训练行为预测器...")
    train_result = arsi.siwm.train_from_history()
    print(f"  状态: {train_result['status']}")
    print(f"  规则数: {train_result.get('rule_count', 0)}")

    # Phase 5: First Governor step
    print("\n[Phase 5] 第一次 Governor 决策...")
    result = arsi.step()
    decision = result["decision"]
    print(f"  动作: {decision['action']}")
    print(f"  理由: {decision['reason'][:80]}")
    print(f"  来源: {decision['source']}")

    # Phase 6: Run a short term
    print("\n[Phase 6] 运行短 term (5 steps)...")
    term_result = arsi.run_term(n_steps=5)
    for s in term_result["steps"]:
        a = s["decision"]["action"]
        src = s["decision"].get("source", "?")
        print(f"  Step {s['step']}: {a} [{src}]")

    # Phase 7: Final stats
    print("\n[Phase 7] 最终状态...")
    final = arsi.get_stats()
    print(json.dumps({k: v for k, v in final.items() if k != "iron_laws"},
                      indent=2, ensure_ascii=False))

    # Summary
    print("\n" + "=" * 60)
    print("冷启动完成！ARSI 已就绪。")
    print("=" * 60)
    print("\n下一步：")
    print("  1. python scripts/run_term.py --steps 10  # 运行完整 term")
    print("  2. python scripts/human_cli.py            # 人类监督界面")
    print("  3. python scripts/demo_llm.py             # LLM 集成演示")

    arsi.close()


if __name__ == "__main__":
    main()
