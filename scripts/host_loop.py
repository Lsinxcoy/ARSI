r"""Real host loop — multi-agent protocol wired to live MiMo / Hermes / SYNTHEX.

Each cycle:
  1. Pull REAL traces from host adapters (no synthetic hosts)
  2. Ingest into ARSI
  3. Register hosts + dispatch evidence-based tasks
  4. Execute REAL side effects:
       - Hermes: EmpowermentApplier writes skills under AppData/.../hermes/skills
       - MiMo:   append feedback under ~/.local/share/mimocode/memory/arsi_feedback
       - SYNTHEX: read-only inventory + optional guidance file under its docs/
  5. Report only what evidence supports (file exists / byte delta / ingest count)

Usage:
    set PYTHONPATH=E:/ARSI/src
    python scripts/host_loop.py
    python scripts/host_loop.py --cycles 2
    python scripts/host_loop.py --http   # also exercise HTTP ma endpoints
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.adapters.hermes_adapter import HermesAdapter
from arsi.adapters.hermes_deep_adapter import HermesDeepAdapter
from arsi.adapters.mimo_deep_extractor import MiMoDeepExtractor
from arsi.adapters.mimo_extractor import MiMoSessionExtractor
from arsi.adapters.synthex_adapter import SynthexAdapter
from arsi.adapters.bidirectional_interface import ARSIInterface
from arsi.core import ARSI
from arsi.empowerment.applier import EmpowermentApplier
from arsi.empowerment.dimensions import DimensionOrchestrator
from arsi.foundation.paths import archive_dir, write_json_once
from arsi.multiagent import MultiAgentOrchestrator

HOSTS = {
    "hermes": {"role": "tool_runner", "capabilities": ["skills", "tools", "msar"]},
    "mimo-desktop": {"role": "worker", "capabilities": ["sessions", "memory", "desktop"]},
    "synthex-mothernest": {"role": "researcher", "capabilities": ["mechanisms", "wal", "state"]},
}

MIMO_FEEDBACK = Path.home() / ".local" / "share" / "mimocode" / "memory" / "arsi_feedback" / "latest_suggestions.md"
SYNTHEx_GUIDANCE = Path(r"E:\SYNTHEX Autopoiesis\docs\arsi_host_loop_guidance.md")


def _file_size(p: Path) -> int:
    try:
        return p.stat().st_size if p.exists() else 0
    except Exception:
        return 0


def pull_real_traces(arsi: ARSI) -> dict:
    stats = {"hermes": 0, "mimo": 0, "synthex": 0, "details": {}}

    try:
        basic = HermesAdapter().extract_traces()
        deep = HermesDeepAdapter().extract_traces(limit_sessions=30, limit_messages=200)
        hermes = basic + deep
        for t in hermes:
            arsi.ingest_trace(
                agent_id="hermes",
                action=t.get("action", "hermes_unknown"),
                outcome=t.get("outcome", "unknown"),
                effect=float(t.get("effect", 0.4) or 0.4),
                params=t.get("params", {}),
            )
        stats["hermes"] = len(hermes)
        stats["details"]["hermes_deep"] = HermesDeepAdapter().get_stats() if hasattr(HermesDeepAdapter(), "get_stats") else {}
    except Exception as e:
        stats["details"]["hermes_err"] = str(e)

    try:
        basic = MiMoSessionExtractor().extract_traces()
        deep = MiMoDeepExtractor().extract_traces()
        mimo = basic + deep
        for t in mimo:
            arsi.ingest_trace(
                agent_id="mimo-desktop",
                action=t.get("action", "mimo_unknown"),
                outcome=t.get("outcome", "unknown"),
                effect=float(t.get("effect", 0.4) or 0.4),
                params=t.get("params", {}),
            )
        stats["mimo"] = len(mimo)
    except Exception as e:
        stats["details"]["mimo_err"] = str(e)

    try:
        sx = SynthexAdapter().extract_traces()
        for t in sx:
            arsi.ingest_trace(
                agent_id="synthex-mothernest",
                action=t.get("action", "synthex_unknown"),
                outcome=t.get("outcome", "unknown"),
                effect=float(t.get("effect", 0.4) or 0.4),
                params=t.get("params", {}),
            )
        stats["synthex"] = len(sx)
        home = Path(r"E:\SYNTHEX Autopoiesis")
        stats["details"]["synthex_home_exists"] = home.exists()
        stats["details"]["synthex_state_files"] = len(list((home / "state").glob("*"))) if (home / "state").exists() else 0
    except Exception as e:
        stats["details"]["synthex_err"] = str(e)

    return stats


def execute_hermes(arsi: ARSI, orchestrator: MultiAgentOrchestrator, task_id: str, brief_id: str) -> dict:
    """Real side effect: dimension analysis + skill write to Hermes skills dir."""
    before = {}
    skills_dir = EmpowermentApplier._get_hermes_skills_dir()
    for name in ("arsi-calibration", "arsi-metacognition", "arsi-environment"):
        before[name] = _file_size(skills_dir / name / "SKILL.md")

    traces = arsi.store.get_recent_traces(n=40, agent_id="hermes") or arsi.store.get_recent_traces(n=40)
    state = arsi.siwm.refresh_state()
    orch = DimensionOrchestrator(arsi.store, llm=None)
    dim_results = {}
    try:
        dim_results = orch.run_all(traces, state)
    except Exception as e:
        dim_results = {"_error": {"error": str(e)}}

    applier = EmpowermentApplier(arsi.store)
    try:
        applier.bind_arsi(arsi)
    except Exception:
        pass
    apply_res = applier.apply_all(dim_results, agent_id="hermes")

    after = {}
    written = []
    for name, bsz in before.items():
        asz = _file_size(skills_dir / name / "SKILL.md")
        after[name] = asz
        if asz > bsz or (asz > 0 and bsz == 0):
            written.append(str(skills_dir / name / "SKILL.md"))

    # Real evidence scoring
    success = bool(written) or apply_res.get("applied", 0) > 0
    effect = 0.0
    if written:
        effect = min(1.0, 0.4 + 0.2 * len(written))
    elif apply_res.get("applied", 0) > 0:
        effect = 0.35
    elif traces:
        effect = 0.15  # analysis ran but no write gap
        success = False
    else:
        success = False

    result = orchestrator.submit_result(
        agent_id="hermes",
        task_id=task_id,
        task_description=f"empowerment apply: {list(dim_results.keys())[:4]}",
        outcome="success" if success else ("partial" if traces else "failure"),
        effect=effect,
        skills_used=[Path(p).parent.name for p in written],
        notes=json.dumps({
            "skills_dir": str(skills_dir),
            "skills_written": written,
            "applied": apply_res.get("applied", 0),
            "before": before,
            "after": after,
            "trace_sample": len(traces),
        }, ensure_ascii=False),
    )
    return {"host": "hermes", "success": success, "effect": effect, "written": written, "result": result}


def execute_mimo(arsi: ARSI, orchestrator: MultiAgentOrchestrator, task_id: str) -> dict:
    """Real side effect: write MiMo feedback file for next session."""
    before = _file_size(MIMO_FEEDBACK)
    state = arsi.siwm.refresh_state()
    advice = arsi.iwm.governor_advice(state) if getattr(arsi, "iwm", None) else {}
    frontier = arsi.iwm.frontier.exploit_bias() if getattr(arsi, "iwm", None) else []
    try:
        from arsi.iwm.host_strategy import build_host_strategy
        strategy = build_host_strategy(arsi, agent_id="mimo-desktop")
        strat_block = "\n".join(strategy.as_structured_block())
        focus = strategy.focus
        conf = strategy.confidence
    except Exception:
        strat_block = ""
        focus = "execute_with_evidence"
        conf = advice.get("self_trust") or 0.5
    line = (
        f"\n## host_loop {datetime.now().isoformat()}\n"
        f"- eta={state.eta:.3f} layer1_holdout={getattr(arsi.siwm, 'last_holdout_accuracy', 0)}\n"
        f"- iwm_forbid_dream={advice.get('forbid_default_dream')} "
        f"prefer_learn={advice.get('prefer_learn')}\n"
        f"- memory_trust={advice.get('memory_trust')} focus={focus} conf={conf:.2f}\n"
        f"- frontier_exploit={frontier[:5]}\n"
        f"{strat_block}\n"
        f"- 建议：按 IWM Strategy focus 执行；Receipt 必须 grounded\n"
    )
    MIMO_FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    existing = MIMO_FEEDBACK.read_text(encoding="utf-8") if MIMO_FEEDBACK.exists() else "# ARSI 赋能建议\n"
    body = existing + line
    try:
        from arsi.foundation.evidence_receipt import fusion_write_and_verify
        fusion = fusion_write_and_verify(MIMO_FEEDBACK, body, receipt_kind="mimo_feedback")
        success = bool(fusion.get("write_ok"))
        effect = 0.5 if success else 0.1
        notes_extra = fusion
    except Exception:
        MIMO_FEEDBACK.write_text(body, encoding="utf-8")
        after = _file_size(MIMO_FEEDBACK)
        success = after > before and after > 0
        effect = 0.5 if success else 0.0
        notes_extra = {"bytes_before": before, "bytes_after": after}
    after = _file_size(MIMO_FEEDBACK)
    result = orchestrator.submit_result(
        agent_id="mimo-desktop",
        task_id=task_id,
        task_description="write ARSI feedback for next MiMo session",
        outcome="success" if success else "failure",
        effect=effect,
        notes=json.dumps({
            "feedback_path": str(MIMO_FEEDBACK),
            "bytes_before": before,
            "bytes_after": after,
            "delta": after - before,
        }, ensure_ascii=False),
    )
    return {"host": "mimo-desktop", "success": success, "effect": effect, "path": str(MIMO_FEEDBACK), "result": result}


def execute_synthex(arsi: ARSI, orchestrator: MultiAgentOrchestrator, task_id: str) -> dict:
    """Real side effect: write guidance doc into SYNTHEX docs/ (not mothernest core)."""
    home = Path(r"E:\SYNTHEX Autopoiesis")
    exists = home.exists()
    guidance = (
        f"# ARSI host-loop guidance\n\n"
        f"Generated: {datetime.now().isoformat()}\n\n"
        f"Source: ARSI multi-agent protocol (read-only toward mothernest core).\n\n"
        f"- Pool/eta/organ snapshot is ARSI-side; do not treat as mothernest verified claims.\n"
        f"- Prefer external effect anchors over constant self-verified scores.\n"
        f"- Multi-agent: synthex remains research observer unless human wires BodyContract.\n"
    )
    written = False
    path = SYNThEX_GUIDANCE if False else Path(r"E:\SYNTHEX Autopoiesis\docs\arsi_host_loop_guidance.md")
    if exists:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(guidance, encoding="utf-8")
            written = path.exists() and path.stat().st_size > 0
        except Exception:
            written = False

    # Inventory evidence
    state_files = list((home / "state").glob("*")) if (home / "state").exists() else []
    success = written
    effect = 0.4 if written else (0.2 if exists else 0.0)
    outcome = "success" if success else ("partial" if exists else "failure")
    result = orchestrator.submit_result(
        agent_id="synthex-mothernest",
        task_id=task_id,
        task_description="write ARSI guidance into SYNTHEX docs (core untouched)",
        outcome=outcome,
        effect=effect,
        notes=json.dumps({
            "home_exists": exists,
            "guidance_path": str(path),
            "written": written,
            "state_file_count": len(state_files),
        }, ensure_ascii=False),
    )
    return {"host": "synthex-mothernest", "success": success, "effect": effect, "path": str(path), "result": result}


def http_roundtrip(port: int = 9300) -> dict:
    """Optional: exercise live HTTP multi-agent endpoints."""
    import urllib.request

    def post(path: str, payload: dict):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())

    def get(path: str):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
            return json.loads(resp.read().decode())

    out = {"ok": False}
    try:
        out["register"] = post("/arsi/ma/register", {
            "agent_id": "hermes-http",
            "role": HOSTS["hermes"]["role"],
            "capabilities": HOSTS["hermes"]["capabilities"],
        })
        d = post("/arsi/ma/dispatch", {"agent_id": "hermes-http", "task": "real host HTTP calibration"})
        out["dispatch"] = d
        if d.get("dispatched"):
            out["result"] = post("/arsi/ma/result", {
                "agent_id": "hermes-http",
                "task_id": d["task_id"],
                "task": "real host HTTP calibration",
                "outcome": "success",
                "effect": 0.3,
                "notes": "http_smoke",
            })
        out["health"] = get("/arsi/ma/health")
        out["ok"] = True
    except Exception as e:
        out["error"] = str(e)
    return out


def run_cycle(arsi: ARSI, orchestrator: MultiAgentOrchestrator, cycle_id: int) -> dict:
    print(f"\n===== HOST LOOP cycle {cycle_id} =====")
    ingest = pull_real_traces(arsi)
    print("ingest_real", json.dumps(ingest, ensure_ascii=False))

    # layer1 + IWM bind from real data
    train = arsi.siwm.train_from_history()
    if getattr(arsi, "iwm", None) is not None:
        arsi.iwm.observe_layer1_holdout(float(train.get("holdout_accuracy") or 0.0))
    print("layer1", train)

    # Ensure registered
    for agent_id, meta in HOSTS.items():
        if agent_id not in orchestrator.agents:
            orchestrator.register(agent_id, role=meta["role"], capabilities=meta["capabilities"])

    # Dispatch evidence-based tasks
    dispatched = {}
    for agent_id in HOSTS:
        task = {
            "hermes": "apply empowerment skills from real Hermes traces",
            "mimo-desktop": "write ARSI feedback for next MiMo session",
            "synthex-mothernest": "write ARSI guidance into SYNTHEX docs",
        }[agent_id]
        d = orchestrator.dispatch(agent_id, task)
        dispatched[agent_id] = d
        print("dispatch", agent_id, d.get("dispatched"), d.get("task_id"), d.get("reason", ""))

    # Execute REAL host side effects
    execs = {}
    if dispatched.get("hermes", {}).get("dispatched"):
        execs["hermes"] = execute_hermes(
            arsi, orchestrator,
            dispatched["hermes"]["task_id"],
            dispatched["hermes"].get("brief_id", ""),
        )
        print("exec_hermes", execs["hermes"]["success"], execs["hermes"]["effect"], execs["hermes"].get("written"))
    if dispatched.get("mimo-desktop", {}).get("dispatched"):
        execs["mimo-desktop"] = execute_mimo(arsi, orchestrator, dispatched["mimo-desktop"]["task_id"])
        print("exec_mimo", execs["mimo-desktop"]["success"], execs["mimo-desktop"]["effect"])
    if dispatched.get("synthex-mothernest", {}).get("dispatched"):
        execs["synthex-mothernest"] = execute_synthex(
            arsi, orchestrator, dispatched["synthex-mothernest"]["task_id"]
        )
        print("exec_synthex", execs["synthex-mothernest"]["success"], execs["synthex-mothernest"]["effect"])

    stats = arsi.get_stats()
    report = {
        "cycle_id": cycle_id,
        "timestamp": datetime.now().isoformat(),
        "ingest": ingest,
        "layer1": train,
        "dispatched": dispatched,
        "executions": {
            k: {
                "success": v["success"],
                "effect": v["effect"],
                "written": v.get("written"),
                "path": v.get("path"),
                "agent_after": v.get("result", {}).get("agent"),
                "effect_anchor": (v.get("result") or {}).get("effect_anchor", {}).get("claim"),
            }
            for k, v in execs.items()
        },
        "multi_agent": orchestrator.health(),
        "arsi_stats": {
            "trace_count": stats.get("trace_count"),
            "experience_count": stats.get("experience_count"),
            "eta": stats.get("eta"),
            "layer1": stats.get("layer1"),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=9300)
    args = parser.parse_args()

    # Real host env
    os.environ.setdefault("PYTHONPATH", r"E:\ARSI\src")
    arsi = ARSI.from_config(r"E:\ARSI\config\arsi.yaml")
    if arsi.llm:
        # host loop side-effects do not require LLM
        arsi.llm._client = getattr(arsi.llm, "_client", None)

    interface = ARSIInterface(arsi)
    orchestrator = MultiAgentOrchestrator(arsi=arsi, interface=interface)
    arsi.multi_agent = orchestrator
    arsi.dream.iwm = getattr(arsi, "iwm", None)

    reports = []
    for i in range(1, max(1, args.cycles) + 1):
        reports.append(run_cycle(arsi, orchestrator, i))

    if args.http:
        print("\n===== HTTP multi-agent =====")
        http_res = http_roundtrip(args.port)
        print(json.dumps(http_res, ensure_ascii=False, default=str)[:2000])
        reports.append({"http": http_res})

    out = archive_dir() / "eval" / "host_loop_latest.json"
    write_json_once(out, {"cycles": reports}, writer_id="scripts.host_loop")
    print("\nWROTE", out)

    # Print agent evidence table
    ma = orchestrator.health()["multi_agent"]
    print("AGENTS", json.dumps(ma.get("agents"), ensure_ascii=False, indent=2))
    arsi.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
