"""ARSI ↔ MiMo Continuous Loop — auto trace export + empowerment feedback.

This script creates a persistent feedback loop:
  1. Extract behavior traces from MiMo Desktop sessions
  2. Ingest into ARSI
  3. Run analysis (MindZero, Governor, dimensions)
  4. Generate empowerment suggestions
  5. Write suggestions to a feedback file
  6. MiMo reads feedback at next session start

Usage:
    # Run once (e.g., after a session ends)
    python scripts/continuous_loop.py

    # Run with specific session
    python scripts/continuous_loop.py --session <session_id>

    # Run in watch mode (polls for new sessions)
    python scripts/continuous_loop.py --watch --interval 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.core import ARSI
from arsi.adapters.mimo_extractor import MiMoSessionExtractor
from arsi.adapters.mimo_deep_extractor import MiMoDeepExtractor
from arsi.adapters.hermes_adapter import HermesAdapter
from arsi.adapters.synthex_adapter import SynthexAdapter
from arsi.empowerment.dimensions import DimensionOrchestrator

# Feedback file location (MiMo reads this at session start)
FEEDBACK_DIR = Path.home() / ".local" / "share" / "mimocode" / "memory" / "arsi_feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "latest_suggestions.md"


def run_loop(arsi: ARSI, extractor: MiMoSessionExtractor, session_id: str = None) -> dict:
    """Run one iteration of the continuous loop."""
    print(f"\n{'='*50}")
    print(f"ARSI Continuous Loop — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    # Step 1: Extract traces from MiMo
    print("\n[1/6] 提取行为轨迹...")
    traces = extractor.extract_traces(session_id)
    print(f"  MiMo 基础: {len(traces)} 条轨迹")

    # Step 1a: Deep extraction from MiMo session memory
    deep_extractor = MiMoDeepExtractor()
    deep_traces = deep_extractor.extract_traces()
    print(f"  MiMo 深度: {len(deep_traces)} 条轨迹（任务树+意图+指令+笔记）")
    traces.extend(deep_traces)

    # Step 1b: Extract traces from Hermes
    hermes = HermesAdapter()
    hermes_traces = hermes.extract_traces()
    print(f"  Hermes: {len(hermes_traces)} 条轨迹")
    traces.extend(hermes_traces)

    # Step 1c: Extract traces from SYNTHEX (母巢)
    synthex = SynthexAdapter()
    synthex_traces = synthex.extract_traces()
    print(f"  SYNTHEX(母巢): {len(synthex_traces)} 条轨迹")
    traces.extend(synthex_traces)

    if not traces:
        print("  无新轨迹，跳过")
        return {"status": "no_traces"}

    # Step 2: Ingest into ARSI
    print("\n[2/6] 导入 ARSI...")
    for t in traces:
        source = t.get("params", {}).get("source", "")
        if source.startswith(("mstar", "hermes", "skill")):
            agent_id = "hermes"
        elif source.startswith(("synthex", "mechanism", "wal", "state")):
            agent_id = "synthex-mothernest"
        else:
            agent_id = "mimo-desktop"
        arsi.ingest_trace(
            agent_id=agent_id,
            action=t.get("action", "unknown"),
            outcome=t.get("outcome", "unknown"),
            effect=t.get("effect", 0.5),
            params=t.get("params", {}),
        )
    stats = arsi.get_stats()
    print(f"  总轨迹: {stats['trace_count']}  经验: {stats['experience_count']}")

    # Step 3: Run a few ARSI steps
    print("\n[3/6] 运行 ARSI 分析循环...")
    for i in range(3):
        result = arsi.step()
        d = result["decision"]
        print(f"  Step {result['step']}: {d['action']} [{d['source']}]")

    # Step 4: Dimension analysis
    print("\n[4/6] 维度分析...")
    recent_traces = arsi.store.get_recent_traces(n=50)
    state = arsi.siwm.refresh_state()
    orch = DimensionOrchestrator(arsi.store, llm=arsi.llm)
    dim_results = orch.run_all(recent_traces, state)

    suggestions = []
    for dim_name, result in dim_results.items():
        if "error" in result:
            continue
        sense = result.get("sense", {})
        gen = result.get("generate", {})
        if sense.get("gap_detected"):
            suggestions.append({
                "dimension": dim_name,
                "issue": gen.get("reason", sense.get("reason", "")),
                "recommendation": gen.get("recommendation", gen.get("strategy", "")),
                "action": gen.get("action", ""),
            })
            print(f"  {dim_name}: 检测到缺口 → {gen.get('action', '?')}")

    if not suggestions:
        print("  未检测到明显缺口")

    # Step 5: Generate empowerment feedback
    print("\n[5/6] 生成赋能建议...")
    feedback = generate_feedback(arsi, suggestions, dim_results)

    # Step 6: Write feedback file
    print("\n[6/6] 写入反馈文件...")
    write_feedback(feedback)

    # Summary
    final_stats = arsi.get_stats()
    print(f"\n{'─'*50}")
    print(f"  η: {final_stats['eta']:.4f}")
    print(f"  梦境: {final_stats['dreams']}  预演: {final_stats['pre_enactment_count']}")
    print(f"  建议数: {len(suggestions)}")
    print(f"  反馈文件: {FEEDBACK_FILE}")
    print(f"{'─'*50}")

    return {
        "status": "ok",
        "traces_ingested": len(traces),
        "suggestions_count": len(suggestions),
        "eta": final_stats["eta"],
        "feedback_file": str(FEEDBACK_FILE),
    }


def generate_feedback(arsi: ARSI, suggestions: list, dim_results: dict) -> str:
    """Generate human-readable empowerment feedback."""
    state = arsi.siwm.refresh_state()
    stats = arsi.get_stats()

    lines = [
        "# ARSI 赋能建议",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"系统代数: {state.phi.generation}  η: {state.eta:.4f}",
        f"轨迹: {stats['trace_count']}  经验: {stats['experience_count']}",
        "",
    ]

    if suggestions:
        lines.append("## 检测到的改进点")
        lines.append("")
        for i, s in enumerate(suggestions, 1):
            lines.append(f"### {i}. {s['dimension'].upper()}")
            lines.append(f"- **问题**: {s['issue']}")
            lines.append(f"- **建议**: {s['recommendation']}")
            lines.append(f"- **操作**: {s['action']}")
            lines.append("")
    else:
        lines.append("## 状态良好")
        lines.append("未检测到明显缺口。系统运行正常。")
        lines.append("")

    # Add belief summary
    if state.psi.beliefs:
        lines.append("## 当前信念")
        for b in state.psi.beliefs[:5]:
            lines.append(f"- {b.content} (置信度: {b.confidence:.2f})")
        lines.append("")

    # Add action recommendations
    lines.append("## 下次会话建议")
    if state.eta >= 0.3:
        lines.append("- ⚠️ η 较高，建议先运行 dream 刷新自我模型")
    if stats["trace_count"] > stats["experience_count"] * 3:
        lines.append("- 建议运行 learn 蒸馏未处理的轨迹")
    if not state.psi.beliefs:
        lines.append("- 信念数为 0，建议积累更多行为数据")
    lines.append("")

    return "\n".join(lines)


def write_feedback(content: str) -> None:
    """Write feedback to file for MiMo to read."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_FILE.write_text(content, encoding="utf-8")
    print(f"  已写入: {FEEDBACK_FILE}")


def read_feedback() -> str:
    """Read the latest feedback (for MiMo to call at session start)."""
    if FEEDBACK_FILE.exists():
        return FEEDBACK_FILE.read_text(encoding="utf-8")
    return "无赋能建议（首次运行）"


def _get_source_mtimes() -> dict:
    """Get latest modification times for all data sources."""
    mtimes = {}

    # MiMo sessions
    mimo_base = Path.home() / ".local" / "share" / "mimocode" / "memory" / "sessions"
    if mimo_base.exists():
        latest = 0
        for d in mimo_base.iterdir():
            if d.is_dir():
                for f in d.iterdir():
                    latest = max(latest, f.stat().st_mtime)
        mtimes["mimo"] = latest

    # Hermes
    hermes_paths = [
        Path("C:/Users/41228/hermes-personal-backup/mstar_fitness.db"),
        Path("C:/Users/41228/hermes-personal-backup/failed_trajectories.jsonl"),
        Path("C:/Users/41228/.hermes/skills"),
        Path("C:/Users/41228/.hermes/plans"),
    ]
    latest = 0
    for p in hermes_paths:
        if p.exists():
            if p.is_file():
                latest = max(latest, p.stat().st_mtime)
            else:
                for f in p.rglob("*"):
                    if f.is_file():
                        latest = max(latest, f.stat().st_mtime)
    mtimes["hermes"] = latest

    # SYNTHEX 母巢
    synthex_paths = [
        Path("E:/SYNTHEX Autopoiesis/synthex_main.db"),
        Path("E:/SYNTHEX Autopoiesis/state"),
        Path("E:/SYNTHEX Autopoiesis/wal_logs"),
    ]
    latest = 0
    for p in synthex_paths:
        if p.exists():
            if p.is_file():
                latest = max(latest, p.stat().st_mtime)
            else:
                for f in p.iterdir():
                    if f.is_file():
                        latest = max(latest, f.stat().st_mtime)
    mtimes["synthex"] = latest

    return mtimes


def main():
    parser = argparse.ArgumentParser(description="ARSI ↔ MiMo Continuous Loop")
    parser.add_argument("--session", help="Specific session ID to extract from")
    parser.add_argument("--watch", action="store_true", help="Watch mode (poll for new sessions)")
    parser.add_argument("--interval", type=int, default=300, help="Watch interval in seconds")
    parser.add_argument("--read-feedback", action="store_true", help="Just read latest feedback")
    args = parser.parse_args()

    if args.read_feedback:
        print(read_feedback())
        return

    # Build ARSI
    arsi = ARSI.from_config("config/arsi.yaml")
    extractor = MiMoSessionExtractor()

    print(f"MiMo 记忆路径: {extractor.memory_base}")
    print(f"会话数: {len(extractor.list_sessions())}")

    if args.watch:
        print(f"\n监听模式（每 {args.interval} 秒检查一次）...")
        print(f"监控源: MiMo Desktop, Hermes, SYNTHEX 母巢")

        # Track last modification times for all sources
        last_mtimes = _get_source_mtimes()
        last_session = None

        while True:
            current_mtimes = _get_source_mtimes()
            changed = False

            # Check MiMo sessions
            sessions = extractor.list_sessions()
            if sessions and sessions[0]["session_id"] != last_session:
                last_session = sessions[0]["session_id"]
                changed = True
                print(f"  MiMo: 新会话 {last_session}")

            # Check Hermes and SYNTHEX file changes
            for source, mtime in current_mtimes.items():
                old = last_mtimes.get(source, 0)
                if mtime > old:
                    changed = True
                    print(f"  {source}: 数据已更新")

            if changed:
                run_loop(arsi, extractor, last_session)
                last_mtimes = current_mtimes

            time.sleep(args.interval)
    else:
        run_loop(arsi, extractor, args.session)

    arsi.close()


if __name__ == "__main__":
    main()
