r"""I6 live acceptance — evaluate introspective_v1_live against daemon evidence.

Reads production archive (health jsonl + checkpoint + optional live probe),
applies thick-data gates, writes archive/eval/i6_live_acceptance.json.

Usage:
    set PYTHONPATH=E:/ARSI/src
    E:\ARSI\.venv\Scripts\python.exe scripts/i6_live_acceptance.py
    ... --json
    ... --skip-suite          # do not run pytest suite (faster; claim may downgrade)

Exit codes:
    0  introspective_v1_live
    2  introspective_v1_candidate (gates not all met — expected until thick)
    3  introspective_skeleton (Q1–Q3 hard fail)
    1  runner error
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.foundation.paths import archive_dir, eval_dir
from arsi.foundation.verified import unverified, verified_from_pytest
from arsi.iwm.i6_live import (
    DEFAULT_THRESHOLDS,
    I6LiveFacts,
    evaluate_i6_live,
    facts_from_checkpoint,
    facts_from_health_rows,
    merge_facts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_health_rows(path: Path, tail: int = 40) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows[-tail:]


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_eval_notes(eval_path: Path) -> dict:
    """paired promote / live regression from latest compare_*.json if present."""
    out = {"paired_promote": False, "live_regression": False, "compare": ""}
    if not eval_path.exists():
        return out
    comps = sorted(eval_path.glob("compare_*.json"), key=lambda p: p.name)
    if not comps:
        return out
    latest = comps[-1]
    try:
        d = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return out
    notes = d.get("notes") or {}
    paired = notes.get("paired_ab") or {}
    out["paired_promote"] = bool(paired.get("promote"))
    out["live_regression"] = bool(d.get("live_regression"))
    out["compare"] = latest.name
    return out


def live_q_facts(run_steps: int = 0) -> I6LiveFacts:
    """Q1–Q3 facts from **daemon health** when possible (no production mutation).

    In-process step probe is optional and heavy — only for empty health Q.
    """
    if run_steps <= 0:
        return I6LiveFacts(note="q_probe_skipped_use_health", source="live_q_probe")
    try:
        from arsi.core import ARSI

        arsi = ARSI.from_config(str(PROJECT_ROOT / "config" / "arsi.yaml"))
        if getattr(arsi, "llm", None):
            arsi.llm._client = None  # acceptance is offline for LLM
        for _ in range(run_steps):
            arsi.step()
        ok, gate = arsi.iwm.q_gate()
        advice = arsi.iwm.governor_advice()
        cal = arsi.iwm.calibrate()
        organs = advice.get("organ_trust") or {}
        measured = sum(1 for v in organs.values() if isinstance(v, (int, float)) and v > 0.05)
        return I6LiveFacts(
            q_gate_ok=bool(ok),
            q_failed=list(gate.get("failed") or []),
            q_pass={k: bool(v.get("pass")) for k, v in (gate.get("scores") or {}).items()},
            organ_trust=dict(organs),
            organ_measured_frac=round(measured / max(1, len(organs)), 4) if organs else 0.0,
            calibration_n=int(cal.get("n") or 0),
            self_trust=float(advice.get("self_trust") or 0.0),
            degrade_to_baseline=bool(advice.get("degrade_to_baseline")),
            source="live_q_probe",
        )
    except Exception as e:
        return I6LiveFacts(note=f"q_probe_error:{e}", source="live_q_probe")


def q_facts_from_health_rows(rows: list[dict]) -> I6LiveFacts:
    """Extract Q1–Q3 / calibration from daemon-written q_gate without re-stepping."""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    for r in reversed(rows):
        iw = r.get("iwm") or {}
        if isinstance(iw, dict) and isinstance(iw.get("iwm"), dict):
            iw = iw["iwm"]
        iw = iw if isinstance(iw, dict) else {}
        qg = iw.get("q_gate") or r.get("iwm_q_gate") or {}
        if not isinstance(qg, dict) or not qg:
            continue
        scores = qg.get("scores") or {}
        q_pass = {k: bool((scores.get(k) or {}).get("pass")) for k in scores}
        if not q_pass:
            continue
        cal = iw.get("calibration") or scores.get("Q6_calibrate") or {}
        if not isinstance(cal, dict):
            cal = {}
        return I6LiveFacts(
            q_gate_ok=bool(qg.get("failed") is not None and len(qg.get("failed") or []) < 2),
            q_failed=list(qg.get("failed") or []),
            q_pass=q_pass,
            calibration_n=int(cal.get("n") or 0),
            self_trust=float(iw.get("self_trust") or cal.get("self_trust") or 0.0),
            degrade_to_baseline=bool(iw.get("degrade_to_baseline") or cal.get("degrade_to_baseline")),
            source="health_q_gate",
        )
    return I6LiveFacts(note="no_q_gate_in_health", source="health_q_gate")


def host_calibration_facts(rows: list[dict]) -> I6LiveFacts:
    """Use measured host effects as explicit confidence claims (C1-1 t2).

    confidence := host effect (declared), success := host success flag.
    """
    from arsi.iwm.calibrate import IntrospectorCalibrator

    cal = IntrospectorCalibrator(min_samples=5)
    n = 0
    for r in rows or []:
        hl = r.get("host_loop") or {}
        hosts = hl.get("hosts") or {}
        if not isinstance(hosts, dict):
            continue
        for name, h in hosts.items():
            if not isinstance(h, dict) or h.get("effect") is None:
                continue
            conf = float(h.get("effect") or 0.0)
            ok = bool(h.get("success"))
            cal.record(f"host_{name}", used_iwm=True, success=ok, note="host_effect", confidence=conf)
            n += 1
        ma = r.get("multi_agent") or {}
        for name, a in (ma.get("agents") or {}).items():
            if not isinstance(a, dict) or a.get("tasks_completed", 0) <= 0:
                continue
            conf = float(a.get("success_rate") or 0.0)
            ok = conf >= 0.5
            cal.record(f"ma_{name}", used_iwm=True, success=ok, note="ma_success_rate", confidence=conf)
            n += 1
    rel = cal.reliability()
    return I6LiveFacts(
        calibration_n=int(rel.get("n") or 0),
        self_trust=float(cal.self_trust()),
        source="host_effect_calibration",
        note=f"host_rows_scanned n_inputs={n}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="ARSI I6 live acceptance")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-suite", action="store_true")
    parser.add_argument("--skip-q", action="store_true", help="skip in-process Q probe (health q_gate still used)")
    parser.add_argument("--q-steps", type=int, default=0, help=">0 runs heavy production step probe")
    parser.add_argument("--tail", type=int, default=40)
    args = parser.parse_args()

    arch = archive_dir()
    ev = eval_dir()
    health_path = arch / "arsi_health.jsonl"
    ckpt_path = arch / "arsi_checkpoint.json"

    rows = load_health_rows(health_path, tail=args.tail)
    ckpt = load_checkpoint(ckpt_path)
    notes = load_eval_notes(ev)

    f_health = facts_from_health_rows(rows)
    f_ckpt = facts_from_checkpoint(ckpt)
    f_q_health = q_facts_from_health_rows(rows)
    f_q = live_q_facts(run_steps=args.q_steps) if args.q_steps > 0 and not args.skip_q else I6LiveFacts(
        note="q_probe_skipped", source="live_q_probe"
    )

    facts = merge_facts(f_health, f_ckpt, f_q_health, f_q, host_calibration_facts(rows))
    facts.paired_promote = bool(notes.get("paired_promote"))
    facts.live_regression = bool(notes.get("live_regression"))

    if args.skip_suite:
        suite = unverified("i6_live_suite", "skipped_by_flag")
    else:
        suite = verified_from_pytest(
            "i6_live_iwm_suite",
            PROJECT_ROOT,
            pytest_args=["-q", "tests/test_iwm.py", "tests/test_i6_live.py"],
            timeout=180,
        )
    facts.suite_verified = suite.to_dict()

    result = evaluate_i6_live(facts, thresholds=DEFAULT_THRESHOLDS, suite_claim=suite)
    result.facts["eval_notes"] = notes
    result.facts["sources"] = {
        "health_rows": len(rows),
        "checkpoint_tick": ckpt.get("tick"),
        "compare": notes.get("compare"),
    }

    out = ev / "i6_live_acceptance.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        report_path = str(out)
    except Exception as e:
        report_path = f"write_failed:{e}"

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
    else:
        print("I6 LIVE ACCEPTANCE")
        print(f"  claim: {result.claim}  live_v1={result.live_v1}")
        print(f"  gates_ok: {[k for k, v in result.gates.items() if v]}")
        print(f"  gates_fail: {[k for k, v in result.gates.items() if not v]}")
        print(f"  pool={facts.world_pool_size} traces={facts.trace_count} "
              f"cycles={facts.manifest_cycles} v_upd={facts.n_v_updates} "
              f"L1={facts.layer1_holdout:.3f} organs_frac={facts.organ_measured_frac}")
        print(f"  cal_n={facts.calibration_n} self_trust={facts.self_trust:.3f} "
              f"paired={facts.paired_promote} regress={facts.live_regression}")
        print(f"  suite: verified={result.verified.get('verified')} reason={result.verified.get('reason')}")
        print(f"  report: {report_path}")
        print("  boundary: candidate until thick live evidence + verified suite; never claim live on seeds alone")

    if result.claim == "introspective_v1_live":
        return 0
    if result.claim == "introspective_skeleton":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
