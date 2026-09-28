"""P-R3 Multi-timescale control (CTM decay × SAHOO contractive × RSI rounds).

Fast scales must not make promises for slow scales. Each scale has its own
change-rate budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# scale -> max relative change per period (control law)
SCALE_BUDGETS = {
    "tick": 0.05,  # ~minutes: small nudges only
    "sync_short": 0.2,  # 30s half-life
    "sync_med": 0.35,  # 5min
    "sync_long": 0.5,  # 30min
    "term": 0.4,
    "dream_rsi": 0.6,
    "harness_round": 0.5,
    "meta": 0.3,  # Γ may only move Ω slowly
}


@dataclass
class ScaleVerdict:
    scale: str
    allowed: bool
    max_delta: float
    requested: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def change_allowed(scale: str, requested_delta: float) -> ScaleVerdict:
    cap = float(SCALE_BUDGETS.get(scale, 0.25))
    d = abs(float(requested_delta))
    ok = d <= cap
    return ScaleVerdict(
        scale=str(scale),
        allowed=ok,
        max_delta=cap,
        requested=d,
        reason="within_scale_budget" if ok else "exceeds_scale_budget_downshift_scale",
    )


def multiscale_contractive(
    quality_gain: float,
    drift_by_scale: dict[str, float],
) -> dict:
    """SAHOO L_Δ per scale: each scale must be contractive (L<1)."""
    out = {}
    for s, drift in (drift_by_scale or {}).items():
        mu = float(quality_gain)
        ld = (float(drift) / abs(mu)) if mu else float("inf")
        out[s] = {
            "L_delta": round(ld, 6),
            "contractive": ld < 1.0,
            "budget": SCALE_BUDGETS.get(s),
        }
    return {"scales": out, "all_contractive": all(v["contractive"] for v in out.values())}
