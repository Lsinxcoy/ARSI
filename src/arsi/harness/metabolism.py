"""P-R8 Metabolic surface (autopoiesis ontology gap).

Maturana/Varela: living systems have metabolism. ARSI mapping (rumination §3):
  matter flow     = trajectories / effects (new traces are intake)
  metabolic rate  = cost / useful effect (CostLedger)
  membrane        = iron laws (G1–G10) — integrity = zero violations

Health face for armor_health / daemon — diagnostic only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence


@dataclass
class MetabolicSurface:
    throughput: float = 0.0  # new trajectories / window
    metabolic_rate: float = 0.0  # cost units / |useful effect|
    membrane_integrity: float = 1.0  # 1 - violation_rate
    useful_effect: float = 0.0
    cost_units: float = 0.0
    n_new_traces: int = 0
    n_violations: int = 0
    n_actions: int = 0
    alive: bool = True
    note: str = "autopoiesis_metabolism_p_r8"

    def to_dict(self) -> dict:
        return asdict(self)


def useful_effect_sum(effects: Sequence[float]) -> float:
    return sum(max(0.0, float(e or 0.0)) for e in (effects or []))


def metabolic_surface(
    *,
    n_new_traces: int = 0,
    window_actions: int = 0,
    effects: Optional[Sequence[float]] = None,
    cost_units: float = 0.0,
    n_iron_violations: int = 0,
    n_calls: Optional[int] = None,
) -> MetabolicSurface:
    """Compose the three autopoiesis faces.

    throughput = n_new_traces / max(1, window_actions)
    metabolic_rate = cost_units / max(useful_effect, eps)  (lower is thriftier)
    membrane_integrity = 1 - n_violations / max(1, n_calls or window_actions)
    alive = membrane_integrity > 0 and (throughput > 0 or useful_effect > 0)
    """
    acts = max(1, int(window_actions or (n_calls or 1)))
    thr = float(n_new_traces) / acts
    ue = useful_effect_sum(effects)
    rate = float(cost_units) / max(ue, 1e-6)
    denom = max(1, int(n_calls if n_calls is not None else acts))
    integrity = 1.0 - (float(n_iron_violations) / denom)
    integrity = max(0.0, min(1.0, integrity))
    alive = integrity > 0.0 and (thr > 0.0 or ue > 0.0 or n_new_traces > 0)
    return MetabolicSurface(
        throughput=round(thr, 6),
        metabolic_rate=round(rate, 6),
        membrane_integrity=round(integrity, 6),
        useful_effect=round(ue, 6),
        cost_units=round(float(cost_units), 6),
        n_new_traces=int(n_new_traces),
        n_violations=int(n_iron_violations),
        n_actions=acts,
        alive=alive,
    )


def metabolism_from_stats(stats: dict) -> MetabolicSurface:
    """Map ARSI get_stats / CostLedger snapshot into a MetabolicSurface."""
    stats = stats or {}
    cost = stats.get("cost") or stats.get("cost_ledger") or {}
    if isinstance(cost, dict):
        tokens = cost.get("tokens") or {}
        cost_units = float(
            cost.get("total_tokens")
            or ((tokens.get("input") or 0) + (tokens.get("output") or 0))
            or cost.get("llm_calls")
            or 0
        )
    else:
        cost_units = 0.0
    traces = stats.get("new_traces") or stats.get("trace_count") or 0
    window = stats.get("window_actions") or stats.get("action_count") or 20
    effects = stats.get("effects") or stats.get("recent_effects") or []
    violations = stats.get("iron_violations") or stats.get("n_iron_violations") or 0
    return metabolic_surface(
        n_new_traces=int(traces or 0),
        window_actions=int(window or 20),
        effects=effects if isinstance(effects, (list, tuple)) else [],
        cost_units=cost_units,
        n_iron_violations=int(violations or 0),
    )
