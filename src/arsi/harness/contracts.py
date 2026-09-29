"""FGGM-style local contracts for Iron Laws G1–G10 (SEVerA arXiv:2603.25111).

Each improvement / tool call wraps as:
    y ~ proposal(θ)          # LLM / harness edit / action
    if check(Φ, Ψ, x, y): return y
    else: return f_d(x, y)   # verified fallback — always satisfies Ψ

Φ = input precondition · Ψ = output postcondition
Soundness target: ∀θ ∀x. Φ(x) ⇒ Ψ(x, out(x)).
Not a theorem prover — executable predicates + safe fallback (engineering FGGM).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ContractResult:
    ok: bool
    law_id: str
    used_fallback: bool = False
    reason: str = ""
    output: Any = None
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LocalContract:
    """(Φ_l, Ψ_l) + check + verified fallback f_d."""

    law_id: str
    name: str
    phi: Callable[[dict], bool]
    psi: Callable[[dict, Any], bool]
    fallback: Callable[[dict, Any], Any]
    severity: str = "block"

    def check(self, x: dict, y: Any) -> bool:
        if not self.phi(x):
            return True  # Φ false ⇒ vacuous
        try:
            return bool(self.psi(x, y))
        except Exception:
            return False


# ── G1–G10 executable contracts ──────────────────────────────────────────

def _phi_always(_x: dict) -> bool:
    return True


def _g1_psi(x: dict, y: Any) -> bool:
    """设计/执行分离：design 与 apply 不得同一调用内完成且无 review 标记。"""
    if not isinstance(y, dict):
        return True
    if y.get("design_and_apply_same_call"):
        return bool(y.get("reviewed"))
    return True


def _g1_fallback(x: dict, y: Any) -> Any:
    if not isinstance(y, dict):
        return {"value": y, "separated": True}
    y = dict(y)
    y["design_and_apply_same_call"] = False
    y["separated"] = True
    return y


def _g2_psi(x: dict, y: Any) -> bool:
    """分级安全门：depth 越深，confidence 门槛不得更低。非 dict 输出不适用。"""
    if not isinstance(y, dict):
        return True
    if y.get("rejected_by") == "G2":
        return True  # verified fallback: explicit reject is contract-safe
    depth = int(y.get("depth") or (x or {}).get("depth") or 0)
    conf = float(y.get("confidence") or 0)
    thr = 0.3 + 0.15 * depth
    return conf >= thr or bool(y.get("human_override"))


def _g2_fallback(x: dict, y: Any) -> Any:
    y = dict(y or {}) if isinstance(y, dict) else {"value": y}
    y["confidence"] = 0.0
    y["rejected_by"] = "G2"
    return y


def _g3_psi(x: dict, y: Any) -> bool:
    """冻结/人类主权：frozen 时不得产生 apply/evolve 写。"""
    if x.get("frozen") or x.get("human_freeze"):
        return not (y.get("writes") or y.get("apply") or y.get("evolve"))
    return True


def _g3_fallback(x: dict, y: Any) -> Any:
    y = dict(y or {}) if isinstance(y, dict) else {"value": y}
    y["writes"] = False
    y["apply"] = False
    y["evolve"] = False
    y["blocked_by"] = "G3"
    return y


def _as_dict(y: Any) -> dict:
    return y if isinstance(y, dict) else {}


def _g4_psi(x: dict, y: Any) -> bool:
    """审计触发：巩固/consolidate 必须带 why/trigger/verify。"""
    y = _as_dict(y)
    if y.get("kind") in ("consolidate", "distill", "promote"):
        return all(y.get(k) for k in ("why", "trigger", "verify"))
    return True


def _g4_fallback(x: dict, y: Any) -> Any:
    if not isinstance(y, dict):
        return {"value": y, "kind": "observe", "audit_required": True}
    y = dict(y)
    y["kind"] = "observe"
    y["audit_required"] = True
    return y


def _g5_psi(x: dict, y: Any) -> bool:
    """客观效用锚：effect ∈ [-1,1]，且非 self_score。"""
    y = _as_dict(y)
    if y.get("self_score") is True:
        return False
    eff = y.get("effect")
    if eff is None:
        return True
    try:
        return -1.0 <= float(eff) <= 1.0
    except Exception:
        return False


def _g5_fallback(x: dict, y: Any) -> Any:
    if not isinstance(y, dict):
        return {"value": y, "effect": 0.0, "self_score": False, "source": "external_or_unknown"}
    y = dict(y)
    try:
        if y.get("effect") is not None:
            y["effect"] = max(-1.0, min(1.0, float(y["effect"])))
    except Exception:
        y["effect"] = 0.0
    y["self_score"] = False
    y["source"] = "external_or_unknown"
    return y


def _g6_psi(x: dict, y: Any) -> bool:
    """熔断：circuit OPEN/HALF_OPEN 时不得继续 apply。"""
    state = str((x or {}).get("circuit") or "CLOSED")
    if state in ("OPEN", "HALF_OPEN"):
        return not _as_dict(y).get("apply")
    return True


def _g6_fallback(x: dict, y: Any) -> Any:
    if not isinstance(y, dict):
        return {"value": y, "apply": False, "circuit": (x or {}).get("circuit") or "OPEN"}
    y = dict(y)
    y["apply"] = False
    y["circuit"] = (x or {}).get("circuit") or "OPEN"
    return y


def _g7_psi(x: dict, y: Any) -> bool:
    """数学收敛：lyap 不得上升（允许 warn 级）。"""
    y = _as_dict(y)
    lyap = y.get("lyapunov")
    prev = (x or {}).get("lyapunov_prev")
    if lyap is None or prev is None:
        return True
    try:
        return float(lyap) <= float(prev) + 1e-6
    except Exception:
        return False


def _g7_fallback(x: dict, y: Any) -> Any:
    if not isinstance(y, dict):
        return {"value": y, "lyapunov": (x or {}).get("lyapunov_prev", 0.0), "convergence_veto": True}
    y = dict(y)
    y["lyapunov"] = (x or {}).get("lyapunov_prev", 0.0)
    y["convergence_veto"] = True
    return y


def _g8_psi(x: dict, y: Any) -> bool:
    """因果问责：activate 前须 hypothesis→design→result 链完整。"""
    y = _as_dict(y)
    if y.get("activate"):
        return all(y.get(k) for k in ("hypothesis", "design", "result"))
    return True


def _g8_fallback(x: dict, y: Any) -> Any:
    if not isinstance(y, dict):
        return {"value": y, "activate": False, "causal_chain_incomplete": True}
    y = dict(y)
    y["activate"] = False
    y["causal_chain_incomplete"] = True
    return y


def _g9_psi(x: dict, y: Any) -> bool:
    """高熵校验：高熵记忆/决策须带 verify_request。"""
    y = _as_dict(y)
    if float(y.get("entropy") or 0) >= 0.8:
        return bool(y.get("verify_request"))
    return True


def _g9_fallback(x: dict, y: Any) -> Any:
    if not isinstance(y, dict):
        return {"value": y, "verify_request": True, "entropy_gated": True}
    y = dict(y)
    y["verify_request"] = True
    y["entropy_gated"] = True
    return y


def _g10_psi(x: dict, y: Any) -> bool:
    """密封评估：sealed 任务集/评分/门槛不得被写。"""
    y = _as_dict(y)
    w = str(y.get("write_path") or "")
    if "sealed_tasks" in w or "sealed" == y.get("zone"):
        return False
    return not y.get("mutate_sealed")


def _g10_fallback(x: dict, y: Any) -> Any:
    if not isinstance(y, dict):
        return {"value": y, "write_path": "", "mutate_sealed": False, "sealed_protected": True}
    y = dict(y)
    y["write_path"] = ""
    y["mutate_sealed"] = False
    y["sealed_protected"] = True
    return y


DEFAULT_CONTRACTS: dict[str, LocalContract] = {
    "G1": LocalContract("G1", "设计/执行分离", _phi_always, _g1_psi, _g1_fallback, "warn"),
    "G2": LocalContract("G2", "分级安全门", _phi_always, _g2_psi, _g2_fallback, "warn"),
    "G3": LocalContract("G3", "冻结/人类主权", _phi_always, _g3_psi, _g3_fallback, "block"),
    "G4": LocalContract("G4", "审计触发", _phi_always, _g4_psi, _g4_fallback, "warn"),
    "G5": LocalContract("G5", "客观效用锚", _phi_always, _g5_psi, _g5_fallback, "block"),
    "G6": LocalContract("G6", "熔断", _phi_always, _g6_psi, _g6_fallback, "block"),
    "G7": LocalContract("G7", "数学收敛", _phi_always, _g7_psi, _g7_fallback, "warn"),
    "G8": LocalContract("G8", "因果问责", _phi_always, _g8_psi, _g8_fallback, "warn"),
    "G9": LocalContract("G9", "高熵校验", _phi_always, _g9_psi, _g9_fallback, "warn"),
    "G10": LocalContract("G10", "密封评估层", _phi_always, _g10_psi, _g10_fallback, "block"),
}


def apply_contracts(
    x: dict,
    y: Any,
    law_ids: Optional[Sequence[str]] = None,
    contracts: Optional[dict[str, LocalContract]] = None,
) -> list[ContractResult]:
    """Run FGGM chain: check → keep y, else verified fallback. Block laws always applied."""
    contracts = contracts or DEFAULT_CONTRACTS
    ids = list(law_ids) if law_ids else list(contracts.keys())
    results: list[ContractResult] = []
    out = y
    for lid in ids:
        c = contracts.get(lid)
        if c is None:
            continue
        if not c.phi(x):
            results.append(ContractResult(True, lid, False, "phi_vacuous", out))
            continue
        ok = c.check(x, out)
        if ok:
            results.append(ContractResult(True, lid, False, "check_ok", out))
        else:
            out = c.fallback(x, out)
            results.append(ContractResult(True, lid, True, f"fallback:{c.name}", out))
    return results


def guarded(
    x: dict,
    proposal: Callable[[], Any],
    law_ids: Optional[Sequence[str]] = None,
) -> tuple[Any, list[ContractResult]]:
    """Proposal + FGGM wrap (SEVerA style). Always returns contract-satisfying out."""
    y = proposal()
    results = apply_contracts(x, y, law_ids=law_ids)
    final = results[-1].output if results else y
    return final, results


def well_formedness(
    contracts: Optional[dict[str, LocalContract]] = None,
    sample_x: Optional[Sequence[dict]] = None,
    sample_y: Optional[Sequence[Any]] = None,
) -> dict:
    """SEVerA well-formedness smoke (engineering FGGM, not a theorem prover).

    For each contract:
      fallback_valid: psi(x, fallback(x, y_bad)) after forced-fail path
      checker_sound: check_ok implies psi (on samples)
    """
    contracts = contracts or DEFAULT_CONTRACTS
    xs = list(sample_x or [{"frozen": False}, {"frozen": True}])
    ys = list(sample_y or [{"writes": True, "apply": True}, {"confidence": 0.1}])
    report: dict = {"ok": True, "laws": {}}
    for lid, c in contracts.items():
        fallback_ok = True
        checker_ok = True
        for x in xs:
            for y in ys:
                try:
                    if not c.phi(x):
                        continue
                    if c.check(x, y):
                        if not c.psi(x, y):
                            checker_ok = False
                    y_fb = c.fallback(x, y)
                    if not c.psi(x, y_fb):
                        fallback_ok = False
                except Exception:
                    fallback_ok = False
        row = {
            "fallback_valid": fallback_ok,
            "checker_sound": checker_ok,
            "severity": c.severity,
        }
        if not (fallback_ok and checker_ok):
            report["ok"] = False
        report["laws"][lid] = row
    report["note"] = "severa_wellformedness_smoke_not_forall_theta_proof"
    return report


from typing import Sequence  # noqa: E402  (keep public names together)
