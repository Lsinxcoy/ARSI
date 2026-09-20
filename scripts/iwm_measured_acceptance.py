r"""Lightweight: measured host organs → IWM trust impact (no heavy LLM/dim analysis)."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, r"E:\ARSI\src")

from arsi.core import ARSI
from arsi.adapters.bidirectional_interface import ARSIInterface
from arsi.empowerment.applier import EmpowermentApplier
from arsi.foundation.paths import archive_dir, write_json_once
from arsi.meta.eval_loop import run_eval_loop
from arsi.multiagent import MultiAgentOrchestrator
from arsi.adapters.hermes_deep_adapter import HermesDeepAdapter
from arsi.adapters.mimo_deep_extractor import MiMoDeepExtractor
from arsi.adapters.synthex_adapter import SynthexAdapter

SKILLS = EmpowermentApplier._get_hermes_skills_dir()
MIMO_FEEDBACK = Path.home() / ".local" / "share" / "mimocode" / "memory" / "arsi_feedback" / "latest_suggestions.md"
SYNTHEx_GUIDE = Path(r"E:\SYNTHEX Autopoiesis\docs\arsi_host_loop_guidance.md")


def fsz(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def main() -> int:
    arsi = ARSI.from_config(r"E:\ARSI\config\arsi.yaml")
    if arsi.llm:
        arsi.llm._client = None
    interface = ARSIInterface(arsi)
    orch = MultiAgentOrchestrator(arsi=arsi, interface=interface)
    arsi.multi_agent = orch
    arsi.dream.iwm = arsi.iwm
    arsi.governor.iwm = arsi.iwm

    def snap():
        st = arsi.get_stats()
        iw = st.get("iwm") or {}
        inner = iw.get("iwm") if isinstance(iw, dict) and "iwm" in iw else iw
        return {
            "organ_trust": (inner or {}).get("organ_trust"),
            "layer1_holdout": (inner or {}).get("layer1_holdout"),
            "memory_note_hint": (inner or {}).get("organ_trust", {}).get("memory") if isinstance(inner, dict) else None,
            "advice": arsi.iwm.governor_advice(arsi.siwm.get_state()),
        }

    before = snap()

    # light ingest (small)
    n_h = n_m = n_s = 0
    try:
        for t in HermesDeepAdapter().extract_traces(10, 80)[:80]:
            arsi.ingest_trace("hermes", t.get("action", "x"), t.get("outcome", "ok"), 0.5, {})
            n_h += 1
    except Exception as e:
        print("hermes_err", e)
    try:
        for t in MiMoDeepExtractor().extract_traces()[:40]:
            arsi.ingest_trace("mimo-desktop", t.get("action", "x"), t.get("outcome", "ok"), 0.5, {})
            n_m += 1
    except Exception as e:
        print("mimo_err", e)
    try:
        for t in SynthexAdapter().extract_traces()[:40]:
            arsi.ingest_trace("synthex-mothernest", t.get("action", "x"), t.get("outcome", "ok"), 0.5, {})
            n_s += 1
    except Exception as e:
        print("synthex_err", e)

    train = arsi.siwm.train_from_history()
    arsi.iwm.observe_layer1_holdout(float(train.get("holdout_accuracy") or 0.0))
    print("layer1", train)

    # register + 3 real-ish result cycles (file side effects, no heavy analysis)
    HOSTS = {
        "hermes": ("tool_runner", ["skills"]),
        "mimo-desktop": ("worker", ["memory"]),
        "synthex-mothernest": ("researcher", ["state"]),
    }
    for aid, (role, caps) in HOSTS.items():
        orch.register(aid, role=role, capabilities=caps)

    last_execs = {}
    for i in range(3):
        for aid in HOSTS:
            task = f"host evidence round {i+1}"
            d = orch.dispatch(aid, task)
            tid = d.get("task_id") or ""
            if not tid:
                continue
            if aid == "hermes":
                p = SKILLS / "arsi-calibration" / "SKILL.md"
                b = fsz(p)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(f"# arsi-calibration\n\nhost_loop accept r{i+1} {datetime.now().isoformat()}\n", encoding="utf-8")
                a = fsz(p)
                success = a >= b and a > 0
                effect = 0.45 if success else 0.1
            elif aid == "mimo-desktop":
                MIMO_FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
                b = fsz(MIMO_FEEDBACK)
                old = MIMO_FEEDBACK.read_text(encoding="utf-8") if MIMO_FEEDBACK.exists() else "# ARSI\n"
                MIMO_FEEDBACK.write_text(old + f"\n## measured-accept r{i+1}\n", encoding="utf-8")
                a = fsz(MIMO_FEEDBACK)
                success = a > b
                effect = 0.5 if success else 0.0
            else:
                try:
                    SYNTHEx_GUIDE.parent.mkdir(parents=True, exist_ok=True)
                    SYNTHEx_GUIDE.write_text(f"# ARSI guidance r{i+1}\n{datetime.now().isoformat()}\n", encoding="utf-8")
                    success = SYNTHEx_GUIDE.exists()
                    effect = 0.4 if success else 0.0
                except Exception:
                    success, effect = False, 0.0
            r = orch.submit_result(aid, tid, task, "success" if success else "failure", effect=effect)
            last_execs[aid] = {
                "success": success,
                "effect": effect,
                "organ": r.get("agent", {}).get("organ_status"),
                "iwm_bind": r.get("iwm_bind"),
                "anchor_verified": (r.get("effect_anchor") or {}).get("claim", {}).get("verified"),
            }
            # same path host_loop uses for IWM after measured
            if (r.get("agent") or {}).get("organ_status") == "measured":
                arsi.iwm.observe_memory(success, note=f"host:{aid}:r{i+1}:{effect}")

    harvest = arsi.harvest_term_tree()
    eval_res = run_eval_loop(arsi, params=arsi.dream_rsi_params)
    after = snap()

    def delta(b, a):
        bt = b.get("organ_trust") or {}
        at = a.get("organ_trust") or {}
        return {k: {"before": bt.get(k), "after": at.get(k), "changed": bt.get(k) != at.get(k)}
                for k in sorted(set(bt) | set(at))}

    adv_b = before.get("advice") or {}
    adv_a = after.get("advice") or {}
    report = {
        "timestamp": datetime.now().isoformat(),
        "ingest_counts": {"hermes": n_h, "mimo": n_m, "synthex": n_s},
        "layer1": train,
        "last_execs": last_execs,
        "multi_agent": orch.health()["multi_agent"]["agents"],
        "unmeasured": orch.health()["multi_agent"]["unmeasured_agents"],
        "iwm_before": {
            "organ_trust": before.get("organ_trust"),
            "layer1_holdout": before.get("layer1_holdout"),
            "advice_keys": {k: adv_b.get(k) for k in ("self_trust", "forbid_default_dream", "prefer_learn", "unreliable_organs")},
        },
        "iwm_after": {
            "organ_trust": after.get("organ_trust"),
            "layer1_holdout": after.get("layer1_holdout"),
            "advice_keys": {k: adv_a.get(k) for k in ("self_trust", "forbid_default_dream", "prefer_learn", "unreliable_organs")},
        },
        "organ_trust_delta": delta(before, after),
        "advice_changed": {
            k: {"before": adv_b.get(k), "after": adv_a.get(k), "changed": adv_b.get(k) != adv_a.get(k)}
            for k in ("self_trust", "forbid_default_dream", "prefer_learn", "unreliable_organs")
        },
        "harvest": {k: harvest.get(k) for k in ("harvested", "pool_size", "traces_kept", "quality_gate")},
        "eval": {
            "rec": eval_res.recommendation,
            "delta": eval_res.delta_score,
            "paired": (eval_res.notes or {}).get("paired_ab", {}).get("reason"),
        },
        "verdict": {
            "organs_measured": all(v.get("organ") == "measured" for v in last_execs.values()),
            "iwm_bind_seen": any(v.get("iwm_bind") for v in last_execs.values()),
            "anchors_ok": sum(1 for v in last_execs.values() if v.get("anchor_verified")),
            "memory_trust_before": (before.get("organ_trust") or {}).get("memory"),
            "memory_trust_after": (after.get("organ_trust") or {}).get("memory"),
            "behavior_trust_before": (before.get("organ_trust") or {}).get("behavior_predictor"),
            "behavior_trust_after": (after.get("organ_trust") or {}).get("behavior_predictor"),
        },
    }

    out = archive_dir() / "eval" / "iwm_measured_acceptance.json"
    write_json_once(out, report, writer_id="iwm_measured_acceptance")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("WROTE", out)
    arsi.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
