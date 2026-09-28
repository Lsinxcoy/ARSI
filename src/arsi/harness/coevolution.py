"""Co-evolution substrate (arXiv:2608.10299) — stages, Red Queen, anchored meta.

Stage 1 Agent-Agent · Stage 2 Agent-Environment · Stage 3 Meta
Anchored Meta Co-Evolution (ARSI constitution patch):
  Ω (how we evolve) may change via Γ; ρ (what we measure against) may NOT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

STAGE1_AGENT = "agent_agent"
STAGE2_ENV = "agent_environment"
STAGE3_META = "meta_coevolution"


@dataclass
class CoEvoVerdict:
    stage: str
    is_coevolution: bool
    reason: str
    detail: dict = field(default_factory=dict)


def classify_coevolution(
    *,
    n_evolving_units: int,
    both_units_change: bool,
    mutual_pressure: bool,
    env_adapts: bool = False,
    omega_adapts: bool = False,
) -> CoEvoVerdict:
    """Paper §2.2/2.3: co-evolution needs ≥2 units, both change, mutual pressure."""
    if n_evolving_units < 2 or not both_units_change or not mutual_pressure:
        return CoEvoVerdict("none", False, "not_coevolution_single_or_static")
    if omega_adapts:
        return CoEvoVerdict(STAGE3_META, True, "evolution_mechanism_adapts")
    if env_adapts:
        return CoEvoVerdict(STAGE2_ENV, True, "environment_adapts")
    return CoEvoVerdict(STAGE1_AGENT, True, "agent_agent_mutual")


def red_queen_pressure(
    a_scores: Sequence[float],
    b_scores: Sequence[float],
    a_task_diff: Sequence[float],
    b_task_diff: Sequence[float],
) -> dict:
    """Mutual adaptive pressure: when A improves, does B's task get harder?

    Returns coupling strength in [0,1] via |corr(ΔA_score, ΔB_diff)| proxy.
    """
    def _deltas(xs: Sequence[float]) -> list[float]:
        return [float(xs[i + 1]) - float(xs[i]) for i in range(max(0, len(xs) - 1))]

    da, db = _deltas(a_scores), _deltas(b_task_diff)
    n = min(len(da), len(db))
    if n < 2:
        return {"coupling": 0.0, "n": n, "note": "insufficient_pairs"}
    ma, mb = sum(da[:n]) / n, sum(db[:n]) / n
    num = sum((da[i] - ma) * (db[i] - mb) for i in range(n))
    va = sum((da[i] - ma) ** 2 for i in range(n)) or 1e-12
    vb = sum((db[i] - mb) ** 2 for i in range(n)) or 1e-12
    corr = num / ((va * vb) ** 0.5)
    return {
        "coupling": round(abs(corr), 4),
        "signed_corr": round(corr, 4),
        "n": n,
        "red_queen": abs(corr) > 0.3,
        "note": "A_improves_should_raise_B_difficulty",
    }


def anchored_meta_ok(
    gamma_changes_omega: bool,
    gamma_touches_rho: bool,
    rho_frozen: bool = True,
) -> tuple[bool, str]:
    """ARSI constitution patch: Γ may revise Ω, never ρ (iron laws / sealed)."""
    if gamma_touches_rho or not rho_frozen:
        return False, "goal_drift_rho_must_stay_grounded"
    if gamma_changes_omega:
        return True, "anchored_meta_coevolution"
    return True, "no_meta_change"


def task_seed_from_failure_cluster(cluster_label: str, n: int = 3) -> list[dict]:
    """Stage-2 task-space: failure clusters → env-evolution task seeds."""
    return [
        {
            "seed_id": f"TS-{cluster_label}-{i}",
            "from_cluster": cluster_label,
            "kind": "failure_mode_resample",
            "status": "proposed",
        }
        for i in range(max(1, int(n)))
    ]


def coevo_fidelity_check(
    *,
    stage: str = STAGE2_ENV,
    mutual_pressure: bool = True,
    gamma_touches_rho: bool = False,
    rho_frozen: bool = True,
) -> dict:
    """Co-evo + GAI interlock: mutual pressure ok; Γ must not touch ρ."""
    ok_meta, meta_why = anchored_meta_ok(
        gamma_changes_omega=(stage == STAGE3_META),
        gamma_touches_rho=gamma_touches_rho,
        rho_frozen=rho_frozen,
    )
    ok = bool(mutual_pressure) and ok_meta
    return {
        "ok": ok,
        "stage": stage,
        "mutual_pressure": bool(mutual_pressure),
        "anchored_meta": ok_meta,
        "meta_reason": meta_why,
        "note": "coevo_three_stage_anchored_meta",
    }
