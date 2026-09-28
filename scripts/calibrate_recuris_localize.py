"""Recuris localization calibration on real host traces (read-only)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def calibrate_localize(traces, max_n: int = 200) -> dict:
    """Map failing traces → synthetic Γ → localize_failure component histogram."""
    from arsi.harness.skill_trace import TraceStep, build_gamma, localize_failure

    hist: Counter = Counter()
    n_fail = 0
    samples = []
    for t in list(traces or [])[:max_n]:
        t = dict(t or {})
        outcome = str(t.get("outcome") or "").lower()
        if "fail" not in outcome and not t.get("fail_class"):
            continue
        n_fail += 1
        params = t.get("params") or {}
        fc = str(params.get("fail_class") or t.get("fail_class") or "")
        if not fc:
            act = str(t.get("action") or "")
            if "compiled_mech" in act or "compile" in act:
                fc = "compile_other"
            elif "terminal" in act or "tool" in act:
                fc = "tool_error"
            elif "gate" in act:
                fc = "gate_reject"
        ck_ok = "fail" not in outcome
        step = TraceStep(
            w_t={"g0": "pending"},
            skills=list(params.get("skills_used") or []),
            action=str(t.get("action") or ""),
            observation=str(t.get("note") or t.get("error") or t.get("action") or ""),
            w_proposed={"g0": "done" if ck_ok else "pending"},
            checker={
                "accepted": ck_ok,
                "reason": "" if ck_ok else f"env_fail:fail_class={fc or outcome}",
                "fail_class": fc,
            },
            w_next={"g0": "done" if ck_ok else "pending"},
        )
        loc = localize_failure(build_gamma(str(t.get("task_id") or t.get("agent_id") or "t"), [step], 0))
        hist[loc.component] += 1
        if len(samples) < 5:
            samples.append({"action": step.action[:40], "component": loc.component, "reason": loc.reason})
    return {
        "n_fail": n_fail,
        "histogram": dict(hist),
        "samples": samples,
        "note": "recuris_localize_on_real_traces",
    }


def main() -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from arsi.core import ARSI

    arsi = ARSI.from_config(str(Path(__file__).resolve().parents[1] / "config" / "arsi.yaml"))
    if getattr(arsi, "llm", None):
        arsi.llm._client = None
    traces = arsi.store.get_recent_traces(n=300)
    out = calibrate_localize(traces)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
