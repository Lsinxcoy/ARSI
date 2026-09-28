"""Evolver — landscape → Change-Manifest (AEGIS stage 3).

Rule templates first; optional LLM refine. Every candidate must pass
scope fence + critic before status=accepted. **Does not auto-edit core** —
manifests are reviewable artifacts (HarnessX Change-Manifest discipline).
"""
from __future__ import annotations

import json
from typing import Optional

from arsi.harness.gates import critic_check
from arsi.harness.manifest import ChangeManifest
from arsi.harness.taxonomy import is_edit_allowed

# label → (edit_type, dim, module, symbol, summary, diff template)
_RULE_TEMPLATES: dict[str, dict] = {
    "generic_failure": {
        "edit_type": "prompt",
        "dimension": "c2",
        "module": "agent_loop",
        "symbol": "brief.policy",
        "summary": "Require explicit fail_class and recovery step before declaring tool failure",
        "diff": (
            "+ ## Fail-handling\n"
            "+ - On tool failure, set fail_class and attempt one recovery (retry/backoff or alternate tool).\n"
            "+ - Do not repeat the identical failing call without a changed precondition.\n"
        ),
        "inverse_op": "- drop Fail-handling block\n",
    },
    "tool_loop": {
        "edit_type": "processor",
        "dimension": "c4",
        "module": "tool_use",
        "symbol": "empowerment.applier",
        "summary": "Advisory processor: abort identical tool retries after 2 identical errors",
        "diff": (
            "+ # processor: tool_retry_guard\n"
            "+ if last_error == current_error and retry_count >= 2:\n"
            "+     abort_with(fail_class='tool_loop')\n"
        ),
        "inverse_op": "- remove tool_retry_guard\n",
    },
    "invalid_or_compile": {
        "edit_type": "processor",
        "dimension": "c6",
        "module": "task_completion_detection",
        "symbol": "sealed_eval.code_verifier",
        "summary": "Synthex compile retry ring: same candidate_id max 1 repair then abandon",
        "diff": (
            "+ on_compile_fail: fail_class='compile_other' + candidate_id\n"
            "+ same_candidate_id_compile_retries: 1\n"
            "+ on_second_compile_fail_same_id: abandon_and_log (no re-version spam)\n"
            "+ require_compile: true + reject_and_log\n"
        ),
        "inverse_op": "- compile retry ring\n",
    },
    "invalid_or_compile_legacy": {
        "edit_type": "config",
        "dimension": "c6",
        "module": "task_completion_detection",
        "symbol": "sealed_eval.code_verifier",
        "summary": "Require compile check before accepting generated mechanism code",
        "diff": (
            "+ require_compile: true\n"
            "+ on_compile_fail: reject_and_log\n"
        ),
        "inverse_op": "- require_compile: false\n",
    },
    "timeout_or_budget": {
        "edit_type": "config",
        "dimension": "c5",
        "module": "agent_loop",
        "symbol": "daemon.tick",
        "summary": "Cap identical action retries within a tick; escalate to maintain",
        "diff": "+ max_identical_action_retries: 2\n+ on_exceed: action=maintain\n",
        "inverse_op": "- max_identical_action_retries\n",
    },
    "premature_complete": {
        "edit_type": "prompt",
        "dimension": "c6",
        "module": "task_completion_detection",
        "symbol": "verified.claim",
        "summary": "Block task_complete without acceptance checklist evidence",
        "diff": "+ require_acceptance_evidence_before_complete: true\n",
        "inverse_op": "- require_acceptance_evidence_before_complete\n",
    },
    "eval_mismatch": {
        "edit_type": "processor",
        "dimension": "c6",
        "module": "task_completion_detection",
        "symbol": "sealed_eval.evaluator",
        "summary": "Log expected vs actual eval fields on mismatch for digester",
        "diff": "+ on_eval_mismatch: emit_expected_actual_pair\n",
        "inverse_op": "- on_eval_mismatch\n",
    },
    "memory_or_context": {
        "edit_type": "config",
        "dimension": "c3",
        "module": "context_management",
        "symbol": "mnemosyne.core",
        "summary": "Trim oldest non-SELF memories first when context pressure high",
        "diff": "+ overflow_policy: drop_oldest_experience_first\n",
        "inverse_op": "- overflow_policy\n",
    },
    "gate_reject": {
        "edit_type": "processor",
        "dimension": "c6",
        "module": "task_completion_detection",
        "symbol": "verified.claim",
        "summary": "Log gate id + candidate id on reject; one repair pass before abandon",
        "diff": (
            "+ on_gate_reject: emit {gate_id, candidate_id, fail_class='gate_reject'}\n"
            "+ max_gate_repair_passes: 1\n"
        ),
        "inverse_op": "- on_gate_reject\n",
    },
    "tool_error": {
        "edit_type": "tool",
        "dimension": "c4",
        "module": "tool_use",
        "symbol": "empowerment.applier",
        "summary": "Hermes tool_error: tag fail_class + one alternate-tool fallback (terminal/execute_code)",
        "diff": (
            "+ on_tool_error: fail_class='tool_error' tool_id={terminal|execute_code}\n"
            "+ fallback: try_alternate_tool once (execute_code↔terminal)\n"
            "+ no identical failing tool call without changed precondition\n"
            "+ report: fail_class + recovery_attempted + recovery_worked\n"
        ),
        "inverse_op": "- on_tool_error block\n",
    },
    "tool_error_terminal": {
        "edit_type": "processor",
        "dimension": "c4",
        "module": "tool_use",
        "symbol": "empowerment.applier",
        "summary": "Hermes terminal: capture exit_code+stderr_tail; classify timeout|perm|notfound; one retry",
        "diff": (
            "+ on_terminal_fail: params={exit_code, stderr_tail(200), tool_id='terminal'}\n"
            "+ classify: timeout|permission|not_found|other\n"
            "+ one retry with changed cwd/env OR switch execute_code; then abandon\n"
            "+ report fail_class=tool_error_terminal + stderr_tail\n"
        ),
        "inverse_op": "- terminal capture block\n",
    },
    "tool_error_execute": {
        "edit_type": "tool",
        "dimension": "c4",
        "module": "tool_use",
        "symbol": "empowerment.applier",
        "summary": "Hermes execute_code: one syntax fix then fallback terminal; capture exception type",
        "diff": (
            "+ on_execute_code_fail: params={exc_type, traceback_tail}\n"
            "+ one_syntax_fix_then_terminal_fallback\n"
            "+ report fail_class=tool_error_execute\n"
        ),
        "inverse_op": "- execute_code capture block\n",
    },
    "script_error": {
        "edit_type": "processor",
        "dimension": "c4",
        "module": "tool_use",
        "symbol": "empowerment.applier",
        "summary": "stdin/script errors: capture traceback; one isolated retry; then abandon",
        "diff": (
            "+ on_script_fail: capture traceback_tail(300) + file_line\n"
            "+ max_isolated_script_retries: 1\n"
            "+ second fail → abandon_and_log fail_class=script_error\n"
            "+ never SUCCESS without acceptance evidence\n"
        ),
        "inverse_op": "- script_error handler\n",
    },
    "path_escape": {
        "edit_type": "tool",
        "dimension": "c4",
        "module": "tool_use",
        "symbol": "empowerment.applier",
        "summary": "Unicode path escape: normalize via pathlib/raw string before call",
        "diff": (
            "+ before_tool_call: pathlib.Path(raw) normalize backslashes\n"
            "+ on_path_escape: fail_class=path_escape + one corrected retry\n"
        ),
        "inverse_op": "- path normalize block\n",
    },
    "type_error": {
        "edit_type": "processor",
        "dimension": "c4",
        "module": "tool_use",
        "symbol": "empowerment.applier",
        "summary": "TypeError/AttributeError: capture exc type+line; one guarded retry",
        "diff": (
            "+ on_type_error: params={exc_type, line}\n"
            "+ one_guarded_retry then abandon_and_log\n"
        ),
        "inverse_op": "- type_error handler\n",
    },
    "runtime_error": {
        "edit_type": "processor",
        "dimension": "c4",
        "module": "agent_loop",
        "symbol": "daemon.tick",
        "summary": "Capture stack + fail_class=runtime_error; isolate one retry",
        "diff": "+ on_exception: fail_class='runtime_error' + capture_stack\n+ max_isolated_retries: 1\n",
        "inverse_op": "- on_exception\n",
    },
    "resource_limit": {
        "edit_type": "control",
        "dimension": "c5",
        "module": "agent_loop",
        "symbol": "daemon.tick",
        "summary": "Downshift parallel batch when resource_limit hit",
        "diff": "+ on_resource_limit: fail_class='resource_limit' + parallelism=max(1, k//2)\n",
        "inverse_op": "- on_resource_limit\n",
    },
}


def _canon_label(label: str) -> str:
    """Map fine labels onto template keys."""
    lab = str(label or "generic_failure")
    if lab in _RULE_TEMPLATES:
        return lab
    aliases = {
        "tool_error": "tool_error",
        "tool_error_terminal": "tool_error_terminal",
        "tool_error_execute": "tool_error_execute",
        "script_error": "script_error",
        "path_escape": "path_escape",
        "gate_reject": "gate_reject",
        "synthex_gate": "gate_reject",
        "runtime_error": "runtime_error",
        "exception": "runtime_error",
        "resource_limit": "resource_limit",
        "compile": "invalid_or_compile",
        "invalid": "invalid_or_compile",
        "compile_retry": "invalid_or_compile",
        "timeout": "timeout_or_budget",
    }
    for k, v in aliases.items():
        if k in lab:
            return v
    return "generic_failure"


class Evolver:
    def __init__(self, llm=None, bmin: int = 1, bmax: int = 3, T: int = 10):
        self.llm = llm
        self.bmin = bmin
        self.bmax = bmax
        self.T = T
        # P-b: strategy-arm bandit over edit types (AIDE85)
        try:
            from arsi.harness.strategy_bandit import EDIT_TYPE_ARMS, StrategyBandit

            self.bandit = StrategyBandit(arms=EDIT_TYPE_ARMS)
        except Exception:
            self.bandit = None

    def propose_from_landscape(
        self,
        landscape: dict,
        max_k: int = 3,
        t_round: int = 0,
        ledger=None,
        stalled: bool = False,
    ) -> list[ChangeManifest]:
        from arsi.harness.edit_budget import edit_budget

        budget = edit_budget(t_round, self.T, bmin=self.bmin, bmax=self.bmax)
        recs = list((landscape or {}).get("recommendations") or [])
        # RRSI: stall reserves m_draft for unexercised components
        if stalled and ledger is not None:
            unex = [c for c in (landscape.get("untried_edit_types") or [])]
            if unex:
                recs = [{"cluster": "explore", "label": "generic_failure", "n": 0,
                         "modules": [], "prefer_edit_type": unex[0]}] + recs
        out: list[ChangeManifest] = []
        # P0-2 dry-run dedup: load prior accepted labels/types
        prior = []
        try:
            from arsi.foundation.paths import archive_dir
            from arsi.harness.manifest import ManifestStore

            prior = ManifestStore(archive_dir() / "harness").list()
        except Exception:
            prior = []
        prior_pairs = {
            (str((m.get("evidence") or [""])[-1]), str(m.get("edit_type") or ""))
            for m in prior
        }
        for rec in recs[: max(1, int(max_k))]:
            label = str(rec.get("label") or "generic_failure")
            # skip if this cluster label already has a manifest of the preferred type
            prefer = rec.get("prefer_edit_type")
            if prefer and any(label in a or a.endswith(str(rec.get("cluster") or "")) for a, b in prior_pairs if b == prefer):
                # rotate to a different edit type
                alts = [t for t in ("prompt", "processor", "tool", "config", "control") if t != prefer]
                rec = dict(rec)
                rec["prefer_edit_type"] = alts[hash(label) % len(alts)] if alts else prefer
            tmpl = dict(_RULE_TEMPLATES.get(label) or _RULE_TEMPLATES.get(_canon_label(label)) or _RULE_TEMPLATES["generic_failure"])
            # P-b: bandit picks the edit-type arm first (diversity lever)
            if self.bandit is not None:
                arm = self.bandit.select()
                if arm != tmpl["edit_type"]:
                    tmpl["edit_type"] = arm
                    tmpl["summary"] = f"[{arm}] {tmpl['summary']}"
            # honor planner untried lever if it maps to a known edit type
            prefer_type = rec.get("prefer_edit_type")
            if prefer_type and prefer_type != tmpl["edit_type"] and prefer_type in (
                "prompt",
                "processor",
                "tool",
                "config",
                "control",
            ):
                tmpl["edit_type"] = prefer_type
                tmpl["summary"] = f"[{prefer_type}] {tmpl['summary']}"
            m = ChangeManifest(
                module=tmpl["module"],
                dimension=tmpl.get("dimension") or "",
                symbol=tmpl["symbol"],
                edit_type=tmpl["edit_type"],
                summary=f"{tmpl['summary']} (cluster={rec.get('cluster')}, n={rec.get('n')})",
                diff=tmpl["diff"],
                inverse_op=tmpl["inverse_op"],
                evidence=list(rec.get("modules") or []) + [str(rec.get("cluster") or "")],
                variant_id="default",
            )
            # RRSI L0: cap independently attributable edits per candidate
            if len(out) >= budget:
                break
            self._gate(m)
            if m.status == "proposed" and self.llm is not None:
                self._llm_refine(m, rec)
                self._gate(m)
            out.append(m)
        return out

    def _gate(self, m: ChangeManifest) -> None:
        ok, reason = is_edit_allowed(m.dimension, m.module, m.symbol)
        if not ok:
            m.status = "rejected"
            m.reject_reason = reason
            return
        # P-R9: Evolver may use semantics, but only after leakage screen
        try:
            from arsi.harness.guidance import screen_guidance

            gv = screen_guidance("evolver_manifest", f"{m.summary}\n{m.diff}")
            if not gv.allowed:
                m.status = "rejected"
                m.reject_reason = f"guidance:{gv.reason}"
                return
        except Exception:
            pass
        crit = critic_check(m.diff, summary=m.summary)
        if not crit.ok:
            m.status = "rejected"
            m.reject_reason = f"{crit.gate}:{crit.reason}"
            return
        m.status = "accepted"

    def _llm_refine(self, m: ChangeManifest, rec: dict) -> None:
        """Optional LLM polish of summary/diff — never unlocks frozen dims."""
        try:
            prompt = (
                "Refine this harness Change-Manifest. Reply JSON with keys summary,diff,inverse_op.\n"
                f"cluster={rec}\nmanifest={json.dumps(m.to_dict(), ensure_ascii=False)[:1500]}\n"
                "Constraints: no secrets, no gold answers, no hardcoded success, keep inverse_op."
            )
            resp = self.llm.chat(prompt) if hasattr(self.llm, "chat") else None
            text = getattr(resp, "content", None) or (resp if isinstance(resp, str) else "")
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(text[start : end + 1])
                m.summary = str(data.get("summary") or m.summary)[:300]
                m.diff = str(data.get("diff") or m.diff)
                m.inverse_op = str(data.get("inverse_op") or m.inverse_op)
        except Exception:
            pass
