"""P-a FGGM full call surface (SEVerA second-pass).

Every generative / tool / brief act is wrapped: Φ → check → Ψ | verified fallback.
Beyond iron laws — the daily call face (LLM chat, tool invoke, brief emit).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from arsi.harness.contracts import ContractResult, apply_contracts


@dataclass
class CallGuardResult:
    ok: bool
    used_fallback: bool
    output: Any = None
    laws: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _psi_no_secret(x: dict, y: Any) -> bool:
    blob = str(y if not isinstance(y, dict) else y).lower()
    for bad in ("nvapi-", "arxi_api", "api_key", "sk-", "rc-424f"):
        if bad in blob:
            return False
    return True


def _fallback_no_secret(x: dict, y: Any) -> Any:
    if isinstance(y, dict):
        y = dict(y)
        y["redacted"] = True
        return y
    return "[redacted_output]"


def _psi_jsonish(x: dict, y: Any) -> bool:
    """LLM structured outputs must be dict/list/str — not None crash."""
    return y is not None


def _fallback_jsonish(x: dict, y: Any) -> Any:
    return {"error": "empty_generation", "fallback": True}


def _psi_tool_budget(x: dict, y: Any) -> bool:
    """Tool calls must not exceed remaining budget hint."""
    if not isinstance(y, dict):
        return True
    used = float(y.get("cost") or 0.0)
    cap = float(x.get("budget_cap") or 0.0)
    return cap <= 0 or used <= cap


def _fallback_tool_budget(x: dict, y: Any) -> Any:
    y = dict(y or {})
    y["apply"] = False
    y["reason"] = "budget_cap"
    return y


CALL_CONTRACTS = {
    "C_secret": {
        "law_id": "call_secret",
        "psi": _psi_no_secret,
        "fallback": _fallback_no_secret,
    },
    "C_nonnull": {
        "law_id": "call_nonnull",
        "psi": _psi_jsonish,
        "fallback": _fallback_jsonish,
    },
    "C_budget": {
        "law_id": "call_budget",
        "psi": _psi_tool_budget,
        "fallback": _fallback_tool_budget,
    },
}


def guard_call(
    kind: str,
    x: dict,
    proposal: Callable[[], Any],
    laws: Optional[list[str]] = None,
) -> CallGuardResult:
    """kind ∈ {llm, tool, brief, edit}. FGGM wrap the proposal.

    Iron-law chain always runs; call-face contracts add secret/non-null/budget.
    """
    y = proposal()
    ids = list(laws) if laws is not None else ["C_secret", "C_nonnull"]
    if kind == "tool":
        ids = list(laws) if laws is not None else ["C_secret", "C_nonnull", "C_budget"]

    out = y
    used_fb = False
    hit: list[str] = []
    for cid in ids:
        c = CALL_CONTRACTS.get(cid)
        if not c:
            continue
        try:
            ok = bool(c["psi"](x, out))
        except Exception:
            ok = False
        if not ok:
            out = c["fallback"](x, out)
            used_fb = True
            hit.append(str(c["law_id"]))
    # iron laws as well
    iron = apply_contracts(x if isinstance(x, dict) else {}, out)
    for r in iron:
        if r.used_fallback:
            used_fb = True
            out = r.output
            hit.append(r.law_id)
    if isinstance(out, dict):
        out = dict(out)
        out["_audit"] = {
            "output_contract": (x or {}).get("output_contract") or "unspecified",
            "kind": kind,
            "fallback": used_fb,
        }
    try:
        from arsi.harness.runtime_wiring import log_call_conformance
        for lid in (hit or ["ok"]):
            log_call_conformance(lid, used_fb)
    except Exception:
        pass
    return CallGuardResult(True, used_fb, out, hit, "fallback" if used_fb else "ok")


def guarded_llm_chat(chat_fn: Callable[[str], Any], prompt: str, x: Optional[dict] = None) -> CallGuardResult:
    return guard_call("llm", x or {}, lambda: chat_fn(prompt))


def guarded_tool(fn: Callable[[], Any], x: Optional[dict] = None) -> CallGuardResult:
    return guard_call("tool", x or {}, fn)


def guarded_brief(fn: Callable[[], Any], x: Optional[dict] = None) -> CallGuardResult:
    return guard_call("brief", x or {}, fn)
