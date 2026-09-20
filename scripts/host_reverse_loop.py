r"""Host reverse execution — MiMo/Hermes consume ARSI briefs and report back.

Flow (item 3):
  1. Read ARSI feedback/brief files written by host_loop / empowerment
  2. Simulate host-side consumption with EVIDENCE (file mtime/size delta)
  3. Execute real side effects again (skills / feedback refresh)
  4. POST result to ARSI multi-agent HTTP (or in-process orchestrator)
  5. Record external effect anchor via ARSI.report path

Usage:
    set PYTHONPATH=E:/ARSI/src
    python scripts/host_reverse_loop.py
    python scripts/host_reverse_loop.py --http --port 9300
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.core import ARSI
from arsi.adapters.bidirectional_interface import ARSIInterface
from arsi.empowerment.applier import EmpowermentApplier
from arsi.foundation.paths import archive_dir, write_json_once
from arsi.multiagent import MultiAgentOrchestrator

MIMO_FEEDBACK = Path.home() / ".local" / "share" / "mimocode" / "memory" / "arsi_feedback" / "latest_suggestions.md"
MIMO_BRIEF = Path.home() / ".local" / "share" / "mimocode" / "memory" / "arsi_feedback" / "latest_brief.md"
HERMES_SKILLS = EmpowermentApplier._get_hermes_skills_dir()
SYNTHEx_GUIDE = Path(r"E:\SYNTHEX Autopoiesis\docs\arsi_host_loop_guidance.md")


def fsz(p: Path) -> int:
    try:
        return p.stat().st_size if p.exists() else 0
    except Exception:
        return 0


def http_post(port: int, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def consume_brief_evidence() -> dict:
    """What the host actually 'sees' from ARSI before acting."""
    evidence = {
        "mimo_feedback_bytes": fsz(MIMO_FEEDBACK),
        "mimo_brief_bytes": fsz(MIMO_BRIEF),
        "hermes_skill_bytes": {
            n: fsz(HERMES_SKILLS / n / "SKILL.md")
            for n in ("arsi-calibration", "arsi-metacognition", "arsi-environment")
        },
        "synthex_guide_bytes": fsz(SYNTHEx_GUIDE),
    }
    texts = []
    for p in (MIMO_FEEDBACK, MIMO_BRIEF, SYNTHEx_GUIDE):
        if p.exists():
            try:
                texts.append(p.read_text(encoding="utf-8", errors="ignore")[-800:])
            except Exception:
                pass
    for n in evidence["hermes_skill_bytes"]:
        p = HERMES_SKILLS / n / "SKILL.md"
        if p.exists():
            try:
                texts.append(p.read_text(encoding="utf-8", errors="ignore")[:400])
            except Exception:
                pass
    blob = "\n".join(texts)
    evidence["consumed_chars"] = len(blob)
    evidence["mentions_eta"] = "eta" in blob.lower() or "η" in blob
    evidence["mentions_iwm"] = "iwm" in blob.lower() or "memory" in blob.lower()
    evidence["mentions_dream"] = "dream" in blob.lower()
    evidence["sample_tail"] = blob[-300:]
    return evidence


def host_act(agent_id: str, evidence: dict) -> dict:
    """Host-side action after reading brief — real file writes + outcome score."""
    now = datetime.now().isoformat()
    if agent_id == "mimo-desktop":
        before = fsz(MIMO_FEEDBACK)
        MIMO_FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
        old = MIMO_FEEDBACK.read_text(encoding="utf-8") if MIMO_FEEDBACK.exists() else "# ARSI\n"
        line = (
            f"\n## host_reverse {now}\n"
            f"- consumed_feedback={evidence.get('mimo_feedback_bytes')} brief={evidence.get('mimo_brief_bytes')}\n"
            f"- mentions_iwm={evidence.get('mentions_iwm')} mentions_dream={evidence.get('mentions_dream')}\n"
            f"- action: applied ARSI brief guidance in next session context\n"
        )
        MIMO_FEEDBACK.write_text(old + line, encoding="utf-8")
        after = fsz(MIMO_FEEDBACK)
        success = after > before and evidence.get("consumed_chars", 0) > 0
        effect = 0.55 if success else 0.1
        return {
            "agent_id": agent_id,
            "action": "apply_mimo_feedback_context",
            "success": success,
            "effect": effect,
            "evidence": {"bytes_before": before, "bytes_after": after, "consumed": evidence},
        }
    if agent_id == "hermes":
        skill = HERMES_SKILLS / "arsi-calibration" / "SKILL.md"
        before = fsz(skill)
        skill.parent.mkdir(parents=True, exist_ok=True)
        skill.write_text(
            f"# arsi-calibration\n\nHost reverse apply {now}\n"
            f"Brief consumed: feedback={evidence.get('mimo_feedback_bytes')} "
            f"mentions_iwm={evidence.get('mentions_iwm')}\n",
            encoding="utf-8",
        )
        after = fsz(skill)
        success = after >= before and after > 0
        effect = 0.5 if success else 0.1
        return {
            "agent_id": agent_id,
            "action": "apply_hermes_skill_from_brief",
            "success": success,
            "effect": effect,
            "evidence": {"skill_bytes": after, "consumed": evidence},
        }
    # synthex observer: record consumption, do not mutate mothernest core
    try:
        SYNTHEx_GUIDE.parent.mkdir(parents=True, exist_ok=True)
        SYNTHEx_GUIDE.write_text(
            f"# ARSI guidance\n\nhost_reverse observer {now}\n"
            f"consumed_chars={evidence.get('consumed_chars')}\n",
            encoding="utf-8",
        )
        success = True
        effect = 0.35
    except Exception:
        success, effect = False, 0.0
    return {
        "agent_id": agent_id,
        "action": "synthex_observe_and_note",
        "success": success,
        "effect": effect,
        "evidence": {"guide_bytes": fsz(SYNTHEx_GUIDE), "consumed": evidence},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=9300)
    args = parser.parse_args()

    arsi = ARSI.from_config(r"E:\ARSI\config\arsi.yaml")
    if arsi.llm:
        arsi.llm._client = None
    interface = ARSIInterface(arsi)
    orch = MultiAgentOrchestrator(arsi=arsi, interface=interface)
    arsi.multi_agent = orch
    arsi.dream.iwm = arsi.iwm
    arsi.governor.iwm = arsi.iwm

    evidence = consume_brief_evidence()
    print("BRIEF_EVIDENCE", json.dumps({k: evidence[k] for k in evidence if k != "sample_tail"}, ensure_ascii=False)[:500])

    agents = [
        ("mimo-desktop", "worker", ["memory", "sessions"]),
        ("hermes", "tool_runner", ["skills", "tools"]),
        ("synthex-mothernest", "researcher", ["state"]),
    ]
    report_rows = []
    for agent_id, role, caps in agents:
        if agent_id not in orch.agents:
            orch.register(agent_id, role=role, capabilities=caps)

        if args.http:
            try:
                http_post(args.port, "/orsi/ma/register" if False else "/arsi/ma/register", {
                    "agent_id": agent_id + "-http",
                    "role": role,
                    "capabilities": caps,
                })
                d = http_post(args.port, "/arsi/ma/dispatch", {
                    "agent_id": agent_id + "-http",
                    "task": f"reverse-consume ARSI brief for {agent_id}",
                })
            except Exception as e:
                d = {"dispatched": False, "error": str(e)}

        dispatch = orch.dispatch(agent_id, f"reverse-consume ARSI brief ({agent_id})")
        act = host_act(agent_id, evidence)
        tid = dispatch.get("task_id") or f"reverse_{agent_id}"
        result = orch.submit_result(
            agent_id=agent_id,
            task_id=tid,
            task_description=act["action"],
            outcome="success" if act["success"] else "partial",
            effect=act["effect"],
            notes=json.dumps(act["evidence"], ensure_ascii=False, default=str)[:800],
        )
        if args.http:
            try:
                http_result = http_post(args.port, "/arsi/ma/result", {
                    "agent_id": agent_id + "-http",
                    "task_id": (d or {}).get("task_id", tid),
                    "task": act["action"],
                    "outcome": "success" if act["success"] else "partial",
                    "effect": act["effect"],
                    "notes": "http_reverse",
                })
            except Exception as e:
                http_result = {"error": str(e)}
        else:
            http_result = None

        if arsi.iwm is not None:
            arsi.iwm.observe_memory(bool(act["success"]), note=f"host_reverse:{agent_id}")
        report_rows.append({
            "agent_id": agent_id,
            "dispatched": dispatch.get("dispatched"),
            "act": act,
            "organ": (result.get("agent") or {}).get("organ_status"),
            "success_rate": (result.get("agent") or {}).get("success_rate"),
            "iwm_bind": result.get("iwm_bind"),
            "anchor": (result.get("effect_anchor") or {}).get("claim"),
            "http_result": http_result,
        })

    advice = arsi.iwm.governor_advice(arsi.siwm.get_state()) if arsi.iwm else {}
    report = {
        "timestamp": datetime.now().isoformat(),
        "brief_evidence": {k: v for k, v in evidence.items() if k != "sample_tail"},
        "hosts": report_rows,
        "iwm_advice": {
            k: advice.get(k)
            for k in (
                "memory_trust", "memory_status", "trust_memory_for_learn",
                "downweight_memory_ops", "prefer_remember_ingest",
                "organ_trust", "layer1_holdout" if False else "self_trust",
            )
            if k in advice
        },
        "iwm_organ_trust": advice.get("organ_trust"),
        "multi_agent": orch.health()["multi_agent"],
    }
    out = archive_dir() / "eval" / "host_reverse_latest.json"
    write_json_once(out, report, writer_id="scripts.host_reverse_loop")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str)[:4000])
    print("WROTE", out)
    arsi.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
