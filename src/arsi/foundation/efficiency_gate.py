"""Efficiency dual-gate for host empowerment (SoL-Pi B-line).

Rule: harness/empowerment channel changes are accepted only when
  1) capability/host outcome does not regress beyond tolerance
  2) at least one efficiency metric improves

Does NOT change Dream-RSI live_capability D7 semantics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional


@dataclass
class EfficiencyGateResult:
    accepted: bool = False
    capability_ok: bool = False
    efficiency_ok: bool = False
    capability_delta: float = 0.0
    efficiency_delta: float = 0.0
    reason: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)


def dual_gate(
    capability_before: float,
    capability_after: float,
    efficiency_before: float,
    efficiency_after: float,
    capability_tolerance: float = 0.05,
    efficiency_higher_is_better: bool = False,
) -> EfficiencyGateResult:
    """capability: higher better; efficiency default: lower better (cost/elapsed)."""
    cap_delta = float(capability_after) - float(capability_before)
    if efficiency_higher_is_better:
        eff_delta = float(efficiency_after) - float(efficiency_before)
        eff_ok = eff_delta > 1e-6
    else:
        eff_delta = float(efficiency_before) - float(efficiency_after)  # savings
        eff_ok = eff_delta > 1e-6
    cap_ok = cap_delta >= -abs(capability_tolerance)
    accepted = bool(cap_ok and eff_ok)
    if accepted:
        reason = f"pass cap_delta={cap_delta:.4f} eff_savings={eff_delta:.4f}"
    elif not cap_ok:
        reason = f"reject_capability_regression cap_delta={cap_delta:.4f}"
    else:
        reason = f"hold_no_efficiency_gain eff_delta={eff_delta:.4f}"
    return EfficiencyGateResult(
        accepted=accepted,
        capability_ok=cap_ok,
        efficiency_ok=eff_ok,
        capability_delta=round(cap_delta, 6),
        efficiency_delta=round(eff_delta, 6),
        reason=reason,
    )


def host_channel_gate(
    host_success_rate_before: float,
    host_success_rate_after: float,
    token_or_elapsed_before: float,
    token_or_elapsed_after: float,
    tolerance: float = 0.05,
) -> EfficiencyGateResult:
    """Host empowerment channel gate: outcome not worse + cost/time down."""
    return dual_gate(
        capability_before=host_success_rate_before,
        capability_after=host_success_rate_after,
        efficiency_before=token_or_elapsed_before,
        efficiency_after=token_or_elapsed_after,
        capability_tolerance=tolerance,
        efficiency_higher_is_better=False,
    )
