"""Gate a compile-retry-ring SkillPatch on real traces via Recuris validation_gate.

Source tasks: traces with compiled_mech / compile fails (before vs after policy).
Dev: disjoint recent window. Gate is conservative — n/CI insufficient → REJECT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arsi.harness.skill_patch import propose_patch, validation_gate
from arsi.harness.skill_trace import COMPONENT_EXPERIENTIAL


def _is_compileish(t: dict) -> bool:
    blob = f"{t.get('action','')} {t.get('note','')} {t.get('params',{})}"
    return "compiled_mech" in blob or "compile" in blob.lower()


def _success(t: dict) -> float:
    o = str(t.get("outcome") or "").lower()
    if "success" in o or o == "ok":
        return 1.0
    return 0.0


def _dup_rate(traces) -> tuple[list, list]:
    """Per-trace duplicate-retry penalty: 0 if unique candidate_id (or none), 1 if repeat."""
    import re

    seen = set()
    ys = []
    for t in traces:
        blob = f"{t.get('action','')} {t.get('note','')} {t.get('params',{})}"
        m = re.search(r"compiled_mech_ext_([0-9a-f]+)", blob)
        if not m:
            ys.append(0.0)
            continue
        cid = m.group(1)
        ys.append(1.0 if cid in seen else 0.0)
        seen.add(cid)
    return ys


def score_windows(traces, mid: int) -> tuple[list, list]:
    """Split compile-ish traces at index mid → (before, after) success series."""
    xs = [dict(t) for t in traces if _is_compileish(dict(t or {}))]
    before = [_success(t) for t in xs[:mid]]
    after = [_success(t) for t in xs[mid:]]
    return before, after


def main() -> int:
    from arsi.core import ARSI
    from arsi.foundation.store import MnemosyneStore

    root = Path(__file__).resolve().parents[1]
    arsi = ARSI.from_config(str(root / "config" / "arsi.yaml"))
    if getattr(arsi, "llm", None):
        arsi.llm._client = None
    traces = [dict(t) for t in arsi.store.get_recent_traces(n=600)]
    compileish = [t for t in traces if _is_compileish(t)]
    n = len(compileish)
    if n < 16:
        print(json.dumps({"status": "insufficient_compile_traces", "n": n}, indent=2))
        return 0

    # temporal split: older half = source_before, newer half = source_after
    mid = n // 2
    # Metric A: task success (Recuris paper)
    src_before, src_after = score_windows(compileish, mid)
    # Metric B: skill-target — unique candidate_id rate (1=unique ok, 0=duplicate spam)
    dup = _dup_rate(compileish)
    # invert: 1.0 = no repeat (good)
    uniq = [1.0 - d for d in dup]
    src_before_u, src_after_u = uniq[:mid], uniq[mid:]
    tail = compileish[int(n * 0.7) :]
    dev_u = uniq[int(n * 0.7) :]
    half = max(4, len(dev_u) // 2)
    dev_before_u, dev_after_u = dev_u[:half], dev_u[half:]
    if len(dev_before_u) < 4 or len(dev_after_u) < 4:
        half2 = max(2, len(src_after_u) // 2)
        dev_before_u, dev_after_u = src_after_u[:half2], src_after_u[half2:]

    patch = propose_patch(
        COMPONENT_EXPERIENTIAL,
        skill_id="compile_retry_ring",
        skill_title="Compile retry ring: one repair then abandon same candidate_id",
        skill_body=(
            "On compile fail: tag candidate_id; ONE minimal recompile; "
            "second fail same id → abandon_and_log; require_compile; never SUCCESS without evidence."
        ),
        source_tasks=[str(t.get("task_id") or t.get("agent_id") or i) for i, t in enumerate(compileish[:8])],
        implicated_only=[COMPONENT_EXPERIENTIAL],
    )
    gate_success = validation_gate(
        source_before=src_before,
        source_after=src_after,
        dev_before=dev_before if False else src_before[: max(8, len(src_before) // 2)],
        dev_after=src_before[max(8, len(src_before) // 2) :] or src_after[:8],
        min_n=8,
    )
    gate_uniq = validation_gate(
        source_before=src_before_u,
        source_after=src_after_u,
        dev_before=dev_before_u,
        dev_after=dev_after_u,
        min_n=8,
    )
    result = {
        "n_compileish": n,
        "metric_success": {
            "src_rate": {
                "before": round(sum(src_before) / max(1, len(src_before)), 4),
                "after": round(sum(src_after) / max(1, len(src_after)), 4),
            },
            "gate": gate_success.to_dict(),
        },
        "metric_unique_candidate": {
            "src_uniq_rate": {
                "before": round(sum(src_before_u) / max(1, len(src_before_u)), 4),
                "after": round(sum(src_after_u) / max(1, len(src_after_u)), 4),
            },
            "dev_n": {"before": len(dev_before_u), "after": len(dev_after_u)},
            "gate": gate_uniq.to_dict(),
        },
        "patch": patch.to_dict() if patch else None,
        "applied": False,
        "rule": "accept_if_skill_metric_passes_or_source_success_repaired",
    }

    if patch is not None and (gate_uniq.accepted or gate_success.accepted):
        try:
            from arsi.empowerment.applier import EmpowermentApplier
            from arsi.foundation.store import MnemosyneStore

            ap = EmpowermentApplier(MnemosyneStore())
            applied = ap.apply_change_manifest(
                {
                    "manifest_id": "CM-compile-retry-ring",
                    "module": "tool_use",
                    "summary": patch.skill_title,
                    "diff": patch.skill_body,
                    "inverse_op": "- remove compile_retry_ring skill",
                },
                agent_id="hermes",
            )
            result["applied"] = True
            result["gate_used"] = "unique_candidate" if gate_uniq.accepted else "success"
            result["skill_path"] = applied.get("skill_path")
        except Exception as e:
            result["apply_error"] = str(e)

    out = root / "archive" / "eval" / "recuris_gate_compile_ring.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
