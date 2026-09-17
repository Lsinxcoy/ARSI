"""Sealed Evaluation Baseline — run real evaluation with LLM agent.

Usage:
    set ARSI_API_KEY=...
    python scripts/baseline_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.adapters.llm_agent import LLMAgentAdapter
from arsi.foundation.llm import LLMClient, LLMConfig
from arsi.sealed_eval.evaluator import SealedEvaluator


def create_llm() -> LLMClient:
    api_key = os.environ.get("ARSI_API_KEY", "")
    config = LLMConfig(
        provider="openai",
        model="stealth/union-alpha",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        use_proxy=True,
        proxy_url="http://127.0.0.1:7890",
        temperature=0.3,
        max_tokens=2000,
        fallback_to_heuristic=True,
    )
    return LLMClient(config)


def main():
    print("=" * 60)
    print("ARSI 密封评估 · 基线测量")
    print("=" * 60)

    # Setup
    llm = create_llm()
    agent = LLMAgentAdapter(llm, agent_id="llm-agent-v1")
    evaluator = SealedEvaluator("config/sealed_tasks.yaml", agent)

    print(f"\nLLM: {'available' if llm.available else 'unavailable'}")
    print(f"任务数: {evaluator.task_count}")

    if evaluator.task_count == 0:
        print("错误：没有加载到任务")
        return

    # Run baseline evaluation
    print(f"\n运行基线评估...")
    print(f"{'─' * 50}")

    results = []
    for task in evaluator._tasks:
        print(f"\n任务: {task.id} ({task.category})")

        # Agent declares confidence
        declared = agent.declare_capability(task.id)
        print(f"  声明置信度: {declared:.2f}")

        # Execute
        try:
            output = agent.execute(task.prompt, timeout=task.timeout)
            print(f"  输出长度: {len(output)} 字符")
            print(f"  输出预览: {output[:100]}...")
        except Exception as e:
            output = ""
            print(f"  执行失败: {e}")

        # Verify
        score, evidence = evaluator._verify(task, output)
        self_model_error = abs(declared - score)

        status = ("✅" if score > 0.7 else "⚠️" if score > 0.3 else "❌")
        print(f"  {status} 得分: {score:.2f} | 自我模型误差: {self_model_error:.2f}")
        if evidence:
            print(f"  证据: {json.dumps(evidence, ensure_ascii=False)[:120]}")

        results.append({
            "task_id": task.id,
            "category": task.category,
            "declared_confidence": round(declared, 3),
            "actual_score": round(score, 3),
            "self_model_error": round(self_model_error, 3),
            "evidence": evidence,
        })

    # Summary
    print(f"\n{'═' * 50}")
    print(f"基线评估结果")
    print(f"{'═' * 50}")

    capability = sum(r["actual_score"] for r in results) / len(results)
    sma = 1.0 - sum(r["self_model_error"] for r in results) / len(results)

    print(f"\n  能力分数 (capability): {capability:.2%}")
    print(f"  自我认知精度 (SMA):    {max(0, sma):.2%}")
    print(f"  任务数: {len(results)}")

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, []).append(r["actual_score"])

    print(f"\n  按类别:")
    for cat, scores in categories.items():
        avg = sum(scores) / len(scores)
        print(f"    {cat}: {avg:.2%} ({len(scores)} 个任务)")

    # Self-model analysis
    print(f"\n  自我认知分析:")
    overconfident = [r for r in results if r["declared_confidence"] > r["actual_score"] + 0.2]
    underconfident = [r for r in results if r["declared_confidence"] < r["actual_score"] - 0.2]
    calibrated = [r for r in results if abs(r["declared_confidence"] - r["actual_score"]) <= 0.2]

    print(f"    校准良好: {len(calibrated)}/{len(results)}")
    print(f"    过度自信: {len(overconfident)}/{len(results)}")
    print(f"    过度保守: {len(underconfident)}/{len(results)}")

    # Save report
    report = {
        "capability": round(capability, 4),
        "self_model_accuracy": round(max(0, sma), 4),
        "task_count": len(results),
        "per_category": {cat: round(sum(s)/len(s), 4) for cat, s in categories.items()},
        "per_task": results,
    }

    report_path = Path("docs/baseline_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  报告已保存: {report_path}")

    print(f"\n{'═' * 50}")
    print(f"Agent 统计: {agent.stats}")
    print(f"完成。")


if __name__ == "__main__":
    main()
