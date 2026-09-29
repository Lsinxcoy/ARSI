"""Digester — compress trajectories into failure clusters (AEGIS stage 1).

Rule-based first (no LLM required). Prefer explicit `params.fail_class` from the
host report schema; else fall back to pattern labels. Never collapse distinct
hosts/gates into one generic bucket when finer evidence exists.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

# canonical report fail_class → digester label
FAIL_CLASS_TO_LABEL = {
    "tool_loop": "tool_loop",
    "tool_error": "tool_error",
    "timeout": "timeout_or_budget",
    "budget": "timeout_or_budget",
    "invalid_input": "invalid_or_compile",
    "compile_other": "invalid_or_compile",
    "syntax": "invalid_or_compile",
    "runtime_error": "runtime_error",
    "resource_limit": "resource_limit",
    "premature_complete": "premature_complete",
    "eval_mismatch": "eval_mismatch",
    "mismatch": "eval_mismatch",
    "memory": "memory_or_context",
    "context": "memory_or_context",
    "overflow": "memory_or_context",
    "gate_reject": "gate_reject",
    "synthex_gate": "gate_reject",
}


@dataclass
class FailureCluster:
    cluster_id: str
    label: str
    n: int = 0
    task_ids: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    implicated: list = field(default_factory=list)  # module / dim hints
    persistent: bool = False
    hosts: list = field(default_factory=list)
    fail_classes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _params(t: dict) -> dict:
    p = t.get("params")
    return p if isinstance(p, dict) else {}


def _blob(t: dict) -> str:
    p = _params(t)
    parts = [
        str(p.get("fail_class") or t.get("fail_class") or ""),
        str(p.get("recovery_attempted") or ""),
        str(t.get("outcome") or ""),
        str(t.get("action") or ""),
        str(t.get("note") or p.get("notes") or ""),
        str(t.get("error") or ""),
        str(p.get("acceptance_evidence") or ""),
        # host gate ids / evidence fragments (synthex_gate:cand_*)
        " ".join(str(x) for x in (p.get("skills_used") or [])[:6]),
        str(t.get("agent_id") or ""),
    ]
    return " ".join(parts).lower()


def _has(t: dict, key: str) -> bool:
    return key in _blob(t)


def _loopish(t: dict) -> bool:
    b = _blob(t)
    return "loop" in b or "retry" in b or "repeat" in b


def _explicit_label(t: dict) -> Optional[str]:
    p = _params(t)
    fc = str(p.get("fail_class") or t.get("fail_class") or "").strip().lower()
    if not fc or fc in ("ok", "success", "none"):
        return None
    if fc in FAIL_CLASS_TO_LABEL:
        return FAIL_CLASS_TO_LABEL[fc]
    # pass through unknown but non-generic fail_class as its own label
    if fc not in ("unknown", "generic", "generic_failure"):
        return fc[:40]
    return None


_LABEL_RULES: list[tuple[str, callable]] = [
    ("tool_loop", lambda t: _has(t, "tool") and _loopish(t)),
    ("tool_error_terminal", lambda t: _has(t, "terminal")),
    ("tool_error_execute", lambda t: _has(t, "execute_code")),
    ("premature_complete", lambda t: _has(t, "complete") or _has(t, "task_complete")),
    ("timeout_or_budget", lambda t: _has(t, "timeout") or _has(t, "budget")),
    ("invalid_or_compile", lambda t: _has(t, "invalid") or _has(t, "compile") or _has(t, "syntax") or _has(t, "compiled_mech")),
    ("compile_retry_ring", lambda t: _has(t, "compiled_mech") and _loopish(t)),
    ("gate_reject", lambda t: _has(t, "gate") or _has(t, "cand") or _has(t, "winner") or _has(t, "sv ")),
    ("eval_mismatch", lambda t: _has(t, "eval") or _has(t, "assert") or _has(t, "mismatch")),
    ("resource_limit", lambda t: _has(t, "resource") or _has(t, "limit")),
    ("runtime_error", lambda t: _has(t, "runtime") or _has(t, "exception") or _has(t, "traceback")),
    ("memory_or_context", lambda t: _has(t, "memory") or _has(t, "context") or _has(t, "overflow") or _has(t, "session_note")),
    ("tool_error_terminal", lambda t: _has(t, "terminal")),
    ("tool_error_execute", lambda t: _has(t, "execute_code")),
    ("tool_error", lambda t: _has(t, "tool") or _has(t, "skill") or _has(t, "empower")),
    ("task_step_fail", lambda t: _has(t, "task_step") or _has(t, "step_fail") or _has(t, "numpy")),
    ("unknown_action", lambda t: "failure act_" in _blob(t) or _has(t, "unknown action")),
    ("generic_failure", lambda t: True),
]


def _label(t: dict) -> str:
    exp = _explicit_label(t)
    if exp:
        return exp
    for lab, fn in _LABEL_RULES:
        try:
            if fn(t):
                return lab
        except Exception:
            continue
    return "generic_failure"


def _implicated(t: dict) -> list[str]:
    b = _blob(t)
    mods = []
    if "tool" in b or "skill" in b or "empower" in b:
        mods.append("tool_use")
    if "memory" in b or "ingest" in b or "context" in b:
        mods.append("context_management")
    if "timeout" in b or "loop" in b or "control" in b:
        mods.append("agent_loop")
    if "eval" in b or "score" in b or "verif" in b or "gate" in b or "compile" in b:
        mods.append("task_completion_detection")
    if "observ" in b or "health" in b:
        mods.append("observation_management")
    return mods or ["agent_loop"]


class Digester:
    """Group failing traces into recurring behavioral deficiencies."""

    def __init__(self, min_cluster: int = 1):
        self.min_cluster = int(min_cluster)

    def is_failure(self, t: dict) -> bool:
        p = _params(t)
        out = str(t.get("outcome") or "").lower()
        fail = str(p.get("fail_class") or t.get("fail_class") or "").lower()
        if fail and fail not in ("ok", "success", ""):
            return True
        return out in ("failure", "failed", "error", "timeout")

    def digest(
        self,
        traces: Iterable[dict],
        prior_labels: Optional[Counter] = None,
    ) -> list[FailureCluster]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for t in traces:
            t = dict(t or {})
            if not self.is_failure(t):
                continue
            lab = _label(t)
            # P: synthex compile retry ring — same candidate_id over cap
            try:
                from arsi.harness.compile_gate import is_reversion_blocked, note_trace_candidate

                cid, abandon, why = note_trace_candidate(_blob(t))
                if abandon:
                    lab = "compile_retry_ring"
                    t = dict(t)
                    t["fail_class"] = "compile_retry_ring"
                    t["note"] = f"{t.get('note') or ''} {why}".strip()
                blocked, bwhy = is_reversion_blocked(_blob(t))
                if blocked:
                    lab = "compile_retry_ring"
                    t = dict(t)
                    t["fail_class"] = "compile_retry_ring"
                    t["note"] = f"{t.get('note') or ''} {bwhy}".strip()
            except Exception:
                pass
            groups[lab].append(t)

        prior_labels = prior_labels or Counter()
        out: list[FailureCluster] = []
        for i, (lab, items) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
            if len(items) < self.min_cluster:
                continue
            tasks = [str(x.get("task_id") or x.get("trace_id") or x.get("agent_id") or i) for x in items]
            impl: list[str] = []
            hosts: list[str] = []
            fcs: Counter = Counter()
            for x in items:
                for m in _implicated(x):
                    if m not in impl:
                        impl.append(m)
                h = str(x.get("agent_id") or _params(x).get("agent_id") or "")
                if h and h not in hosts:
                    hosts.append(h)
                fc = str(_params(x).get("fail_class") or x.get("fail_class") or "")
                if fc:
                    fcs[fc] += 1
            out.append(
                FailureCluster(
                    cluster_id=f"FC-{lab}-{len(items)}",
                    label=lab,
                    n=len(items),
                    task_ids=tasks[:32],
                    evidence=[_blob(x)[:160] for x in items[:5]],
                    implicated=impl,
                    persistent=bool(prior_labels.get(lab, 0) > 0),
                    hosts=hosts[:8],
                    fail_classes=dict(fcs),
                )
            )
        return out

    @staticmethod
    def label_counts(traces: Iterable[dict]) -> Counter:
        c: Counter = Counter()
        dig = Digester()
        for t in traces:
            t = dict(t or {})
            if dig.is_failure(t):
                c[_label(t)] += 1
        return c
