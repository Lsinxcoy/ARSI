"""RRSI L2 fidelity layer (arXiv:2609.24972) — cost rule, δ, three-track eval.

Paper notes:
  ΔC = (Ĉ(H') - Ĉ(H_t)) / Ĉ(H_t)   relative policy-token cost
  Branch A (ΔS > δ): ΔC ≤ β0 + β1·ΔS   (coding β0=0.10 β1=44.5)
  Branch B (ΔS ≤ δ): ws·ΔS − wc·ΔC + wn·ν_str > 0
  δ from repeated H0 scores only — never holdout/OOD
  Three tracks: evolve | ID held-out | OOD (Table 1 evidence: OOD is the point)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

# Paper Table 5-style defaults (paper_confirmed coding/workspace/eng)
BETA0 = {"coding": 0.10, "workspace": 0.10, "eng": 0.15}
BETA1 = {"coding": 44.5, "workspace": 35.4, "eng": 24.4}
WS = {"coding": 0.0, "workspace": 0.3, "eng": 0.3}  # coding: noise-band score alone insufficient

# K vocabulary (paper §3.3)
COMPONENT_VOCAB = (
    "prompt",
    "control_flow",
    "config",
    "output_plumbing",
    "context_mgmt",
    "client_tool",
    "skill",
    "memory",
    "subagent",
)
STRUCTURAL = frozenset({"client_tool", "skill", "memory", "subagent"})


def relative_cost_delta(c_new: float, c_base: float) -> float:
    """ΔC = (C' − C) / C  (paper Eq. cost)."""
    base = float(c_base)
    if base <= 0:
        return 0.0 if float(c_new) <= 0 else 1.0
    return (float(c_new) - base) / base


def cost_rule_ok(dS: float, dC_rel: float, beta0: float, beta1: float) -> tuple[bool, str]:
    """Branch A: ΔC ≤ β0 + β1·ΔS."""
    limit = float(beta0) + float(beta1) * float(dS)
    ok = float(dC_rel) <= limit + 1e-9
    return ok, ("cost_ok" if ok else f"cost_exceeds_limit:{dC_rel:.4f}>{limit:.4f}")


@dataclass
class ThreeTrackReport:
    evolve_mean: float = 0.0
    id_holdout_mean: float = 0.0
    ood_mean: float = 0.0
    evolve_n: int = 0
    id_n: int = 0
    ood_n: int = 0
    ood_gain_vs_h0: float = 0.0
    note: str = "rrsi_table1_style_tracks"

    def to_dict(self) -> dict:
        return asdict(self)


def three_track_eval(
    evolve_scores: Sequence[float],
    id_scores: Sequence[float] = (),
    ood_scores: Sequence[float] = (),
    h0_ood: Optional[float] = None,
) -> ThreeTrackReport:
    def _m(xs: Sequence[float]) -> float:
        xs = [float(x) for x in (xs or [])]
        return round(sum(xs) / len(xs), 6) if xs else 0.0

    ev, idh, ood = _m(evolve_scores), _m(id_scores), _m(ood_scores)
    gain = round(ood - float(h0_ood), 6) if h0_ood is not None else 0.0
    return ThreeTrackReport(
        evolve_mean=ev,
        id_holdout_mean=idh,
        ood_mean=ood,
        evolve_n=len(list(evolve_scores or [])),
        id_n=len(list(id_scores or [])),
        ood_n=len(list(ood_scores or [])),
        ood_gain_vs_h0=gain,
    )


@dataclass
class RRSIRoundVerdict:
    admissible: bool
    reason: str
    branch: str = ""
    b_t: int = 1
    stalled: bool = False
    m_draft: int = 0
    unexercised: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def rrsi_round(
    *,
    t: int,
    T: int,
    score_hat: float,
    score_now: float,
    score_w_ago: float,
    s_star: float,
    delta: float,
    dS: float,
    c_new: float,
    c_base: float,
    domain: str = "coding",
    components: Sequence[str] = (),
    accepted_counts: Optional[dict] = None,
    all_components: Optional[Sequence[str]] = None,
    diff: str = "",
    summary: str = "",
) -> RRSIRoundVerdict:
    """One RRSI round: b_t + stall U_t + relative cost branch + accept semantics."""
    from arsi.harness.edit_budget import edit_budget, stall_flag
    from arsi.harness.accept import admit, structural_novelty
    from arsi.harness.noise_floor import passes_floor

    b_t = edit_budget(t, T, bmin=1, bmax=4)
    stalled = stall_flag(score_now, score_w_ago, delta)
    unex = sorted(set(all_components or COMPONENT_VOCAB) - set(components or ()))
    m_draft = 1 if stalled else 0

    if not passes_floor(score_hat, s_star, delta):
        return RRSIRoundVerdict(
            False, "floor", "floor", b_t, stalled, m_draft, unex,
            {"score_hat": score_hat, "s_star": s_star, "delta": delta},
        )

    dC_rel = relative_cost_delta(c_new, c_base)
    nu = structural_novelty(components, accepted_counts or {})
    beta0 = BETA0.get(domain, 0.10)
    beta1 = BETA1.get(domain, 35.4)
    ws = WS.get(domain, 0.0)

    if dS > delta:
        ok, why = cost_rule_ok(dS, dC_rel, beta0, beta1)
        if not ok:
            return RRSIRoundVerdict(False, why, "above_delta", b_t, stalled, m_draft, unex, {
                "dS": dS, "dC_rel": dC_rel, "beta0": beta0, "beta1": beta1,
            })
        branch = "above_delta"
    else:
        shaped = ws * float(dS) - 1.0 * float(dC_rel) + 0.1 * float(nu)
        if shaped <= 0:
            return RRSIRoundVerdict(False, "within_band", "within_delta", b_t, stalled, m_draft, unex, {
                "dS": dS, "dC_rel": dC_rel, "nu": nu, "shaped": shaped,
            })
        branch = "within_delta"

    return RRSIRoundVerdict(
        True, "ok", branch, b_t, stalled, m_draft, unex,
        {"dS": dS, "dC_rel": dC_rel, "nu": nu, "domain": domain},
    )


def delta_from_h0(baseline_scores: Sequence[float], method: str = "range") -> float:
    """δ only from repeated unchanged H0 — never holdout/OOD (paper App.D)."""
    from arsi.harness.noise_floor import calibrate_delta

    return calibrate_delta(baseline_scores, method=method)
