"""ARSI Human Supervision CLI — monitor, interact, and override.

Usage:
    set ARSI_API_KEY=sk-...
    python scripts/human_cli.py

Commands:
    stats       — show system stats
    state       — show world state (Φ+Ψ+η)
    step        — run one step
    term [N]    — run a term with N steps (default 10)
    dream       — force dream cycle
    empower ID  — empower an agent
    traces [N]  — show recent traces (default 10)
    beliefs     — show current beliefs
    laws        — show iron laws
    eval        — run evaluation
    help        — show this help
    quit        — exit
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.core import ARSI


def format_dict(d: dict, indent: int = 2) -> str:
    return json.dumps(d, indent=indent, ensure_ascii=False, default=str)


def cmd_stats(arsi: ARSI) -> None:
    stats = arsi.get_stats()
    print("\n── 系统状态 ──")
    for k, v in stats.items():
        if k == "iron_laws":
            print(f"  {k}: {len(v)} laws")
        else:
            print(f"  {k}: {v}")


def cmd_state(arsi: ARSI) -> None:
    state = arsi.siwm.refresh_state()
    print("\n── 联合世界状态 ──")
    print(f"  Φ (物理):")
    print(f"    代数: {state.phi.generation}")
    print(f"    存储: {state.phi.storage_stats}")
    print(f"  Ψ (心智):")
    print(f"    信念数: {len(state.psi.beliefs)}")
    print(f"    情绪: {state.psi.affect}")
    for i, b in enumerate(state.psi.beliefs[:5]):
        print(f"    信念[{i}]: {b.content[:60]} (conf={b.confidence:.2f})")
    print(f"  η (失配度): {state.eta:.4f}")
    depth = arsi.siwm.eta.adaptive_depth()
    print(f"  仿真深度: {depth}")
    if arsi.siwm.eta.should_dream():
        print(f"  ⚠ η 超阈值，建议 dream")


def cmd_step(arsi: ARSI) -> None:
    result = arsi.step()
    print(f"\n── Step {result['step']} ──")
    d = result["decision"]
    print(f"  动作: {d['action']}")
    print(f"  理由: {d['reason'][:100]}")
    print(f"  来源: {d['source']}")
    for a in result["actions"]:
        print(f"  执行: {a.get('type', '?')} — {str(a)[:80]}")


def cmd_term(arsi: ARSI, n: int = 10) -> None:
    print(f"\n── 运行 term ({n} steps) ──")
    result = arsi.run_term(n_steps=n)
    for s in result["steps"]:
        d = s["decision"]
        print(f"  Step {s['step']}: {d['action']} [{d.get('source', '?')}]")
    print(f"\n  评估: {format_dict(result['evaluation'])}")


def cmd_dream(arsi: ARSI) -> None:
    state = arsi.siwm.refresh_state()
    eta_before = state.eta
    new_state = arsi.dream.execute(state)
    print(f"\n── 梦境周期 ──")
    print(f"  η: {eta_before:.4f} → {new_state.eta:.4f}")
    print(f"  信念数: {len(state.psi.beliefs)} → {len(new_state.psi.beliefs)}")


def cmd_empower(arsi: ARSI, agent_id: str) -> None:
    print(f"\n── 赋能 {agent_id} ──")
    result = arsi.empower_agent(agent_id)
    print(f"  诊断维度: {result['diagnosis'].get('dimension', '?')}")
    print(f"  诊断理由: {result['diagnosis'].get('reason', '?')[:100]}")
    print(f"  验证状态: {result['verification']}")


def cmd_traces(arsi: ARSI, n: int = 10) -> None:
    traces = arsi.store.get_recent_traces(n=n)
    print(f"\n── 最近 {len(traces)} 条轨迹 ──")
    for t in traces:
        print(f"  [{t.get('timestamp', '?')[:19]}] {t.get('agent_id', '?')} "
              f"{t.get('action', '?')} → {t.get('outcome', '?')} "
              f"(effect={t.get('effect', 0):.2f})")


def cmd_beliefs(arsi: ARSI) -> None:
    state = arsi.siwm.refresh_state()
    print(f"\n── 当前信念 ({len(state.psi.beliefs)}) ──")
    for i, b in enumerate(state.psi.beliefs):
        print(f"  [{i}] {b.content}")
        print(f"      conf={b.confidence:.2f} source={b.source}")


def cmd_laws(arsi: ARSI) -> None:
    print(f"\n── 铁律 ({len(arsi.iron_laws.law_ids)}) ──")
    for law_id in arsi.iron_laws.law_ids:
        law = arsi.iron_laws.get_law(law_id)
        if law:
            print(f"  {law.id}: {law.name} [{law.severity}]")
            print(f"      {law.description}")


def cmd_eval(arsi: ARSI) -> None:
    result = arsi.run_evaluation()
    print(f"\n── 评估 ──")
    print(format_dict(result))


def main():
    print("=" * 60)
    print("ARSI 人类监督界面")
    print("=" * 60)
    print("输入 help 查看命令列表\n")

    arsi = ARSI.from_config("config/arsi.yaml")

    while True:
        try:
            raw = input("arsi> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd == "quit" or cmd == "exit":
                break
            elif cmd == "help":
                print(__doc__)
            elif cmd == "stats":
                cmd_stats(arsi)
            elif cmd == "state":
                cmd_state(arsi)
            elif cmd == "step":
                cmd_step(arsi)
            elif cmd == "term":
                n = int(args[0]) if args else 10
                cmd_term(arsi, n)
            elif cmd == "dream":
                cmd_dream(arsi)
            elif cmd == "empower":
                agent = args[0] if args else "agent_alpha"
                cmd_empower(arsi, agent)
            elif cmd == "traces":
                n = int(args[0]) if args else 10
                cmd_traces(arsi, n)
            elif cmd == "beliefs":
                cmd_beliefs(arsi)
            elif cmd == "laws":
                cmd_laws(arsi)
            elif cmd == "eval":
                cmd_eval(arsi)
            else:
                print(f"未知命令: {cmd}。输入 help 查看帮助。")
        except Exception as e:
            print(f"错误: {e}")

    arsi.close()


if __name__ == "__main__":
    main()
