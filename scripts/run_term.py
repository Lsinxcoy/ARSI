"""Run a full ARSI improvement term.

Usage:
    set ARSI_API_KEY=sk-...
    python scripts/run_term.py [--steps 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.core import ARSI


def main():
    parser = argparse.ArgumentParser(description="Run an ARSI improvement term")
    parser.add_argument("--config", default="config/arsi.yaml", help="Config file path")
    parser.add_argument("--steps", type=int, default=10, help="Steps per term")
    parser.add_argument("--agent", default="agent_alpha", help="Agent ID for trace simulation")
    parser.add_argument("--simulate", action="store_true", help="Simulate traces before running")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    # Build ARSI
    arsi = ARSI.from_config(args.config)

    if not args.json:
        print("=" * 60)
        print("ARSI Improvement Term")
        print("=" * 60)
        print(f"Config: {args.config}")
        print(f"Steps: {args.steps}")
        print(f"LLM: {'available' if arsi.llm and arsi.llm.available else 'unavailable'}")
        print()

    # Simulate traces if requested
    if args.simulate:
        if not args.json:
            print("Simulating behavior traces...")
        actions = ["learn", "evolve", "reflect", "remember", "maintain"]
        for i in range(20):
            arsi.ingest_trace(
                agent_id=args.agent,
                action=actions[i % len(actions)],
                outcome="success" if i % 3 != 0 else "failure",
                effect=0.3 + (i * 0.03),
            )
        if not args.json:
            print(f"  Ingested 20 traces")
            print()

    # Run term
    if not args.json:
        print("Running term...")
        print()

    result = arsi.run_term(n_steps=args.steps)

    # Output
    if args.json:
        # Clean up non-serializable items
        output = {
            "term_id": result["term_id"],
            "evaluation": result["evaluation"],
            "final_stats": result["final_stats"],
            "step_actions": [
                {"step": s["step"], "action": s["decision"]["action"]}
                for s in result["steps"]
            ],
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print("Steps executed:")
        for s in result["steps"]:
            action = s["decision"]["action"]
            reason = s["decision"]["reason"][:60]
            print(f"  Step {s['step']}: {action} — {reason}")

        print()
        print("Evaluation:")
        eval_r = result["evaluation"]
        print(f"  Capability proxy: {eval_r['capability_proxy']:.2%}")
        print(f"  Self-model η: {eval_r['self_model_eta']:.4f}")
        print(f"  Generation: {eval_r['generation']}")

        print()
        print("Final stats:")
        stats = result["final_stats"]
        for k in ["trace_count", "experience_count", "proxy_count", "eta",
                   "governor_decisions", "empowerments", "dreams", "step_count"]:
            if k in stats:
                print(f"  {k}: {stats[k]}")

        print()
        print("=" * 60)
        print("Term complete.")
        print("=" * 60)

    arsi.close()


if __name__ == "__main__":
    main()
