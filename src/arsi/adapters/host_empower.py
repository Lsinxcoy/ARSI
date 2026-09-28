"""Host empowerment pack — maximize host capability through the protocol.

ARSI exists to serve hosts. The brief face MAY be semantic, readable, and
actionable (P-R9 host boundary). Dual boundaries still hold:
  1) ARSI internal replay_selection remains free of semantic guidance
  2) claims stay negative epistemology (never blind SUCCESS)

Pack contents (max empowerment):
  skill kit · tool routing · active Change-Manifests · fail recovery playbook
  · known drawbacks · continuous-dream what-if · armor/vitals · scaffold level
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

# fail_class → one recovery action (actionable, not philosophy)
FAIL_RECOVERY = {
    "tool_error": [
        "set_fail_class=tool_error + tool_id (terminal|execute_code)",
        "one_backoff_retry OR switch_alternate_tool (execute_code↔terminal)",
        "no_identical_failing_call_without_changed_precondition",
        "report: fail_class + recovery_attempted + recovery_worked",
    ],
    "tool_error_terminal": [
        "MUST capture exit_code + stderr_tail(200) into report.measurements",
        "classify timeout|permission|not_found|resource|other",
        "one retry with changed cwd/env OR execute_code fallback",
        "then abandon; report fail_class=tool_error_terminal",
    ],
    "timeout": [
        "shrink_scope or split step",
        "retry with shorter budget once",
        "report fail_class=timeout + exit_code",
    ],
    "not_found": [
        "verify path/command; list alternatives",
        "switch to alternate tool once",
        "report fail_class=not_found",
    ],
    "script_error": [
        "capture traceback tail",
        "one isolated retry with fixed stdin script",
        "report fail_class=script_error + stderr_tail",
    ],
    "path_escape": [
        "normalize path (raw string / pathlib)",
        "retry once with corrected path",
        "report fail_class=path_escape",
    ],
    "type_error": [
        "capture exception type + line",
        "guard type; one retry",
        "report fail_class=type_error",
    ],
    "tool_error_execute": [
        "capture exc_type + traceback_tail",
        "one syntax fix then terminal fallback",
        "report fail_class=tool_error_execute",
    ],
    "tool_loop": ["abort_identical_call", "change_precondition_or_stop", "set_fail_class"],
    "timeout": ["shrink_scope_or_split", "retry_with_shorter_budget", "set_fail_class"],
    "invalid_input": ["validate_and_repair_args", "ask_for_schema_if_unclear", "set_fail_class"],
    "compile_other": [
        "run_compile_check + tag candidate_id",
        "ONE minimal_fix_then_recompile",
        "same candidate_id second fail → abandon_and_log (no re-version spam)",
        "reject_and_log + never SUCCESS",
    ],
    "compile_retry_ring": [
        "ABANDON candidate_id now",
        "do not re-version same hash",
        "open NEW candidate only with changed design",
        "report fail_class=compile_retry_ring",
    ],
    "runtime_error": ["capture_stack_fail_class", "one_isolated_retry", "escalate_to_maintain"],
    "resource_limit": ["free_or_downgrade_payload", "serialize_parallel_batch", "set_fail_class"],
    "premature_complete": ["attach_acceptance_evidence", "recheck_checklist", "never_label_SUCCESS"],
    # P1-2: gate rejects are the dominant synthex cluster — first-class recovery
    "gate_reject": [
        "log_gate_id_and_candidate_id",
        "one_repair_pass_only",
        "set_fail_class=gate_reject",
        "never_label_SUCCESS_without_acceptance",
    ],
    "synthex_gate": [
        "log_gate_id_and_candidate_id",
        "one_repair_pass_only",
        "set_fail_class=gate_reject",
    ],
    "unknown": ["record_observation_only", "no_success_claim", "set_fail_class"],
}

DEFAULT_FAIL_KEYS = [
    "tool_error",
    "script_error",
    "compile_other",
    "gate_reject",
    "tool_loop",
    "timeout",
    "premature_complete",
]

# default skill kit (host-facing, executable)
DEFAULT_SKILL_KIT = [
    {"skill": "fail-handling", "use_when": "any tool/LLM call fails", "action": "set fail_class + one recovery"},
    {"skill": "acceptance-check", "use_when": "declaring task complete", "action": "require evidence receipt"},
    {"skill": "compile-gate", "use_when": "emitting mechanism code", "action": "compile before accept"},
    {"skill": "bounded-context", "use_when": "history/tool payload huge", "action": "use handle + excerpt"},
    {"skill": "negative-claim", "use_when": "reporting outcome", "action": "no_known_defect only"},
]


@dataclass
class EmpowermentPack:
    skill_kit: list = field(default_factory=list)
    tool_routing: list = field(default_factory=list)
    active_manifests: list = field(default_factory=list)
    fail_recovery: dict = field(default_factory=dict)
    known_drawbacks: list = field(default_factory=list)
    what_if: list = field(default_factory=list)
    armor: dict = field(default_factory=dict)
    scaffold_level: float = 1.0
    preflight: Optional[dict] = None
    note: str = "host_face_semantic_ok_internal_selection_forbid"

    def to_dict(self) -> dict:
        return asdict(self)

    def as_structured_block(self) -> list[str]:
        """Host-facing empowerment block — readable + actionable."""
        lines = ["## Empowerment Pack (max host capability)"]
        if self.skill_kit:
            lines.append("### Skill Kit")
            for s in self.skill_kit:
                if isinstance(s, dict):
                    lines.append(f"- {s.get('skill')}: when={s.get('use_when')} → {s.get('action')}")
                else:
                    lines.append(f"- {s}")
        if self.tool_routing:
            lines.append("### Tool Routing")
            for t in self.tool_routing:
                if isinstance(t, dict):
                    lines.append(f"- {t.get('name')}: {t.get('route')} (priority={t.get('priority', 1)})")
                else:
                    lines.append(f"- {t}")
        if self.active_manifests:
            lines.append("### Active Change-Manifests (structure Ω)")
            for m in self.active_manifests[:8]:
                mid = m.get("manifest_id") or m.get("id") or "?"
                lines.append(f"- {mid} [{m.get('edit_type','?')}] {m.get('summary','')[:100]}")
                if m.get("inverse_op"):
                    lines.append(f"    inverse_op: {str(m.get('inverse_op'))[:80]}")
        if self.fail_recovery:
            lines.append("### Fail Recovery Playbook (executable)")
            for fc, acts in list(self.fail_recovery.items())[:8]:
                lines.append(f"- {fc}: {'; '.join(acts) if isinstance(acts, list) else acts}")
        if self.known_drawbacks:
            lines.append("### Known Drawbacks (do not claim absent)")
            for d in self.known_drawbacks[:8]:
                lines.append(f"- {d}")
        if self.what_if:
            lines.append("### Continuous Dream What-If (planning reference only)")
            for w in self.what_if[:5]:
                lines.append(f"- {w}")
        if self.armor:
            lines.append("### Armor / Vitals (compact)")
            for k, v in list(self.armor.items())[:8]:
                lines.append(f"- {k}: {v}")
        if self.preflight:
            lines.append("### Harness Preflight (read-only)")
            for row in (self.preflight.get("top_clusters") or [])[:4]:
                lines.append(f"- {row.get('label')} n={row.get('n')} hosts={row.get('hosts')}")
            cg = self.preflight.get("compile_gate") or {}
            if cg:
                lines.append(f"- compile_gate: pass={cg.get('n_pass')} reject={cg.get('n_reject')} rate={cg.get('reject_rate')}")
            lines.append(f"- mode: {self.preflight.get('mode')}")
        lines.append(f"- scaffold_level: {self.scaffold_level:.2f} (hint fades as capability rises)")
        lines.append("- claim_rule: no_known_defect / verified_contract / measured only — never is_correct/is_safe")
        return lines


def default_tool_routing(agent_id: str = "") -> list[dict]:
    return [
        {"name": "compile_check", "route": "sealed_eval.code_verifier", "priority": 1},
        {"name": "fail_class_tag", "route": "report.params.fail_class", "priority": 1},
        {"name": "evidence_receipt", "route": "foundation.evidence_receipt", "priority": 2},
        {"name": "history_query", "route": "brief.query_history (structured only)", "priority": 3},
    ]


def fail_recovery_for(fail_classes: Optional[Sequence[str]] = None) -> dict:
    keys = list(fail_classes or DEFAULT_FAIL_KEYS)
    return {k: list(FAIL_RECOVERY.get(k, FAIL_RECOVERY["unknown"])) for k in keys}


def active_manifests_from_store(store=None, limit: int = 8) -> list[dict]:
    if store is None:
        try:
            from arsi.foundation.paths import archive_dir
            from arsi.harness.manifest import ManifestStore

            store = ManifestStore(archive_dir() / "harness")
        except Exception:
            return []
    try:
        rows = store.list(status="accepted") or store.list()
    except Exception:
        return []
    out = []
    for r in rows[-limit:]:
        out.append(
            {
                "manifest_id": r.get("manifest_id"),
                "edit_type": r.get("edit_type"),
                "summary": r.get("summary"),
                "inverse_op": r.get("inverse_op"),
                "status": r.get("status"),
                "dimension": r.get("dimension"),
            }
        )
    return out


def what_if_from_pool(world_pool, max_items: int = 3) -> list[str]:
    """P-R2 continuous dream half-step CFs as planning references (not directives)."""
    try:
        from arsi.harness.pr_wiring import continuous_dream_from_pool

        pack = continuous_dream_from_pool(world_pool, max_worlds=2, steps_between=1)
    except Exception:
        return []
    out = []
    for d in pack.get("dreams") or []:
        scores = d.get("exact_scores") or []
        if len(scores) >= 2:
            delta = scores[-1] - scores[-2]
            out.append(
                f"world={d.get('world_id','?')[:8]} exact_scores={scores[-3:]} "
                f"interp_steps={d.get('n_interp')} cf={d.get('n_counterfactual')} Δ={delta:+.3f}"
            )
        else:
            out.append(
                f"world={d.get('world_id','?')[:8]} nodes_exact={d.get('n_exact')} interp={d.get('n_interp')}"
            )
        if len(out) >= max_items:
            break
    return out


def armor_compact(arsi=None) -> dict:
    try:
        from arsi.harness.runtime_wiring import armor_health

        scores = list(getattr(arsi, "_live_capability_scores", []) or [])[-10:] if arsi else []
        gdi = float(getattr(getattr(arsi, "iwm", None), "last_gdi", 0.0) or 0.0) if arsi else 0.0
        h = armor_health(scores, gdi=gdi)
        return {
            "anchored": h.get("anchored"),
            "overshoot": (h.get("overshoot") or {}).get("steps_past_best"),
            "membrane_integrity": (h.get("metabolism") or {}).get("membrane_integrity"),
            "throughput": (h.get("metabolism") or {}).get("throughput"),
            "harness_monotone_ok": (h.get("harness_monotone") or {}).get("ok"),
        }
    except Exception:
        return {"note": "armor_unavailable"}


def known_drawbacks_from_traces(traces: Sequence[dict], limit: int = 6) -> list[str]:
    """Negative knowledge: patterns of known defects — never claim these absent without check."""
    counts: dict[str, int] = {}
    for t in traces or []:
        fc = str((t or {}).get("fail_class") or (t or {}).get("params", {}).get("fail_class") or "")
        if not fc:
            o = str((t or {}).get("outcome") or "").lower()
            if "fail" in o:
                fc = "generic_failure"
            elif "invalid" in o or "compile" in o:
                fc = "invalid_or_compile"
        if fc:
            counts[fc] = counts.get(fc, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: -x[1])[:limit]
    return [f"{k} n={v}" for k, v in ranked]


def harness_preflight(
    arsi=None,
    agent_id: str = "",
    n_traces: int = 80,
) -> dict:
    """P1-3 read-only harness preflight (never triggers evolve/apply).

    Returns: recent fail clusters for this host + recommended playbook keys
    + compile gate ledger. Safe to call before every task.
    """
    traces = []
    try:
        if arsi is not None:
            traces = arsi.store.get_recent_traces(n=n_traces, agent_id=agent_id) or []
            if agent_id:
                traces = [t for t in traces if str(t.get("agent_id") or "") in ("", agent_id)][:n_traces]
    except Exception:
        traces = []
    dig = None
    try:
        from arsi.harness.digester import Digester
        from arsi.harness.compile_gate import get_compile_ledger

        dig = Digester()
        clusters = dig.digest(traces)
        ledger = get_compile_ledger().to_dict()
    except Exception:
        clusters, ledger = [], {}
    try:
        from arsi.harness.terminal_subtypes import subtype_histogram

        subtypes = subtype_histogram(traces)
    except Exception:
        subtypes = {}
    top = [
        {
            "label": c.label,
            "n": c.n,
            "hosts": list(getattr(c, "hosts", []) or []),
            "fail_classes": dict(getattr(c, "fail_classes", {}) or {}),
            "persistent": c.persistent,
        }
        for c in clusters[:5]
    ]
    playbook_keys = [c.label for c in clusters[:5]] or ["gate_reject", "compile_other"]
    return {
        "available": True,
        "agent_id": agent_id,
        "n_traces": len(traces),
        "n_failures": sum(1 for t in traces if dig.is_failure(t)) if dig is not None else 0,
        "top_clusters": top,
        "recommend_playbook": fail_recovery_for(playbook_keys),
        "compile_gate": ledger,
        "reversion_blocked": __import__(
            "arsi.harness.compile_gate", fromlist=["abandoned_candidates"]
        ).abandoned_candidates()[:12],
        "terminal_capture_required": ["exit_code", "stderr_tail", "stdout_tail"],
        "terminal_subtypes": subtypes,
        "mode": "read_only_no_evolve",
        "note": "p1_3_harness_preflight",
    }


def build_empowerment_pack(
    arsi=None,
    agent_id: str = "",
    fail_classes: Optional[Sequence[str]] = None,
    scaffold_level: float = 1.0,
) -> EmpowermentPack:
    """Assemble max-empowerment host pack."""
    traces = []
    try:
        if arsi is not None:
            traces = arsi.store.get_recent_traces(n=80, agent_id=agent_id) or []
    except Exception:
        traces = []
    pack = EmpowermentPack(
        skill_kit=list(DEFAULT_SKILL_KIT),
        tool_routing=default_tool_routing(agent_id),
        active_manifests=active_manifests_from_store(),
        fail_recovery=fail_recovery_for(fail_classes or DEFAULT_FAIL_KEYS + ["compile_retry_ring"]),
        known_drawbacks=known_drawbacks_from_traces(traces)
        + [
            f"abandoned_candidate:{cid}"
            for cid in (__import__("arsi.harness.compile_gate", fromlist=["abandoned_candidates"]).abandoned_candidates() or [])[:8]
        ],
        what_if=what_if_from_pool(getattr(arsi, "world_pool", None)) if arsi else [],
        armor=armor_compact(arsi),
        scaffold_level=max(0.0, min(1.0, float(scaffold_level))),
        preflight=harness_preflight(arsi, agent_id=agent_id) if arsi is not None else None,
    )
    # W4 hint fade: lower scaffold when live capability is strong
    try:
        scores = list(getattr(arsi, "_live_capability_scores", []) or []) if arsi else []
        if scores:
            from arsi.harness.feedback_evo import hint_fade

            pack.scaffold_level = hint_fade(sum(scores[-5:]) / max(1, len(scores[-5:])), 1.0)
    except Exception:
        pass
    return pack
