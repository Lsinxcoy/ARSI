"""RRSI non-compensatory acceptance (arXiv:2609.24972 §3.3 / App.C.3).

Order matters:
  0) leakage screen BEFORE evaluation
  1) noise floor  Ŝ ≥ S* − δ
  2a) ΔS > δ  → cost rule  ΔC ≤ β0 + β1·ΔS
  2b) ΔS ≤ δ  → shaped   ws·ΔS − wc·ΔC + wn·ν_str > 0
  3) domain guards
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

from arsi.harness.contracts import apply_contracts
from arsi.harness.gates import critic_check
from arsi.harness.noise_floor import passes_floor

# RRSI K_str: novelty only for structural components
STRUCTURAL_COMPONENTS = frozenset({"client_tool", "skill", "memory", "subagent"})


@dataclass
class AcceptVerdict:
    admissible: bool
    reason: str
    branch: str = ""
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def structural_novelty(components: Sequence[str], accepted_counts: dict[str, int]) -> int:
    """ν_t(H') — structural components never seen in a winning edit."""
    n = 0
    for c in set(components or []):
        if c in STRUCTURAL_COMPONENTS and int(accepted_counts.get(c, 0)) == 0:
            n += 1
    return n


def cost_ok(dS: float, dC: float, delta: float, beta0: float, beta1: float) -> bool:
    """Branch A cost rule. dC may be absolute or relative ΔC=(C'-C)/C (RRSI paper)."""
    if dS > delta:
        return dC <= beta0 + beta1 * dS
    return True


def relative_cost(c_new: float, c_base: float) -> float:
    """ΔC relative policy-token cost (paper)."""
    try:
        from arsi.harness.rrsi import relative_cost_delta

        return relative_cost_delta(c_new, c_base)
    except Exception:
        base = float(c_base)
        return ((float(c_new) - base) / base) if base > 0 else 0.0


def within_band_ok(
    dS: float,
    dC: float,
    nu: int,
    ws: float = 0.0,
    wc: float = 1.0,
    wn: float = 0.1,
) -> bool:
    """ws·ΔS − wc·ΔC + wn·ν > 0.  coding-like: ws=0 (score bump alone insufficient)."""
    return (ws * dS) - (wc * dC) + (wn * nu) > 0


def domain_guard(
    *,
    valid_output_rate_before: float,
    valid_output_rate_after: float,
    no_submission_before: float = 0.0,
    no_submission_after: float = 0.0,
    valid_drop_max: float = 0.03,
    no_sub_rise_max: float = 0.02,
) -> tuple[bool, str]:
    """RRSI App.C.3 non-compensatory: primary score cannot compensate execution validity."""
    if valid_output_rate_before - valid_output_rate_after > valid_drop_max:
        return False, "valid_output_dropped"
    if no_submission_after - no_submission_before > no_sub_rise_max:
        return False, "no_submission_risen"
    return True, "ok"


def select_among_admissible(
    candidates: list[dict],
    s_star: float,
) -> tuple[Optional[dict], float]:
    """RRSI final-round: max Ŝ among admissible; else incumbent; update S*.

    candidates: [{admissible, score_hat, ...}]
    """
    adm = [c for c in (candidates or []) if c.get("admissible")]
    if not adm:
        return None, float(s_star)
    best = max(adm, key=lambda c: float(c.get("score_hat") or 0.0))
    new_s = max(float(s_star), float(best.get("score_hat") or 0.0))
    return best, new_s


def admit(
    *,
    diff: str,
    summary: str = "",
    score_hat: float,
    s_star: float,
    delta: float,
    dS: float,
    dC: float,
    components: Sequence[str],
    accepted_counts: dict[str, int],
    beta0: float = 0.10,
    beta1: float = 35.0,
    ws: float = 0.0,
    wc: float = 1.0,
    wn: float = 0.1,
    guards: Optional[Sequence[Callable[..., bool]]] = None,
    detail_extras: Optional[dict] = None,
) -> AcceptVerdict:
    # 0) leakage BEFORE scores are trusted
    leak = critic_check(diff, summary=summary)
    if not leak.ok:
        return AcceptVerdict(False, f"leakage:{leak.reason}", branch="critic")

    # 0b) SEVerA FGGM chain on the proposed edit payload
    payload = {
        "writes": True,
        "apply": True,
        "kind": "promote",
        "effect": dS,
        "write_path": summary,
        "confidence": score_hat,
    }
    cresults = apply_contracts({"frozen": False, "circuit": "CLOSED"}, payload)
    hard = [r for r in cresults if r.used_fallback and r.law_id in ("G3", "G5", "G6", "G10")]
    if hard:
        return AcceptVerdict(
            False,
            "iron_law_guard",
            branch="fggm",
            detail={"laws": [r.law_id for r in hard]},
        )

    # 1) noise-adjusted floor
    if not passes_floor(score_hat, s_star, delta):
        return AcceptVerdict(
            False,
            "floor",
            branch="floor",
            detail={"score_hat": score_hat, "s_star": s_star, "delta": delta},
        )

    nu = structural_novelty(components, accepted_counts)
    if dS > delta:
        if not cost_ok(dS, dC, delta, beta0, beta1):
            return AcceptVerdict(
                False,
                "cost_rule",
                branch="above_delta",
                detail={"dS": dS, "dC": dC, "beta0": beta0, "beta1": beta1},
            )
        branch = "above_delta"
    else:
        if not within_band_ok(dS, dC, nu, ws=ws, wc=wc, wn=wn):
            return AcceptVerdict(
                False,
                "within_band",
                branch="within_delta",
                detail={"dS": dS, "dC": dC, "nu": nu, "ws": ws, "wc": wc, "wn": wn},
            )
        branch = "within_delta"

    extras = detail_extras or {}
    if extras:
        try:
            from arsi.harness.runtime_wiring import admit_extras
            ok_x, why_x, _ = admit_extras(**extras)
            if not ok_x:
                return AcceptVerdict(False, why_x, branch=branch or "extras")
        except Exception:
            pass

    for g in guards or []:
        try:
            if not g():
                return AcceptVerdict(False, "domain_guard", branch=branch)
        except TypeError:
            # guard takes no args in this API
            pass

    return AcceptVerdict(
        True,
        "ok",
        branch=branch,
        detail={"dS": dS, "dC": dC, "nu": nu},
    )
