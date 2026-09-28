"""Wire P-R modules into live loops (dream_rsi / eval_loop / evolver / daemon).

Import-light helpers so core/daemon call one place.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence


def continuous_dream_from_pool(world_pool, max_worlds: int = 3, steps_between: int = 2) -> dict:
    """P-R2: synthesize continuous dreams from newest pool worlds."""
    from arsi.world_model.continuous_dream import synthesize_from_discovery_tree

    worlds = list(getattr(world_pool, "worlds", []) or [])[-max_worlds:]
    dreams = []
    for w in worlds:
        tree = getattr(w, "tree", None) or getattr(w, "_tree", None)
        if tree is None:
            # ReplayWorld keeps nodes in _full; wrap lightly
            full = getattr(w, "_full", None) or {}
            if not full:
                continue
            from arsi.world_model.discovery_tree import DiscoveryTree

            tree = DiscoveryTree()
            try:
                tree.build_from_traces([dict(v) for k, v in full.items() if k != "root"])
            except Exception:
                continue
        dream = synthesize_from_discovery_tree(tree, steps_between=steps_between)
        dreams.append(
            {
                "world_id": getattr(w, "world_id", ""),
                "n_exact": dream.n_exact,
                "n_interp": dream.n_interp,
                "n_counterfactual": dream.n_counterfactual,
                "exact_scores": dream.exact_scores(),
            }
        )
    return {"dreams": dreams, "note": "p_r2_continuous_dream_wired"}


def log_unified_after_dream(
    ledger,
    t: int,
    *,
    policy_id: str = "",
    manifest_id: str = "",
    dS: float = 0.0,
    dC: float = 0.0,
    accepted: bool = False,
    policy_share: float = 0.6,
) -> dict:
    """P-R4: one improvement that may have moved both Ω books."""
    if ledger is None:
        return {"logged": False, "reason": "no_ledger"}
    if policy_id and manifest_id:
        rec = ledger.log_both(
            t,
            f"{policy_id}×{manifest_id}",
            policy_share=policy_share,
            dS=dS,
            dC=dC,
            accepted=accepted,
            hypothesis=f"policy={policy_id}|manifest={manifest_id}",
        )
    elif policy_id:
        rec = ledger.log_dream(t, policy_id, dS=dS, dC=dC, accepted=accepted)
    elif manifest_id:
        rec = ledger.log_manifest(t, manifest_id, dS=dS, dC=dC, accepted=accepted)
    else:
        return {"logged": False, "reason": "no_id"}
    return {"logged": True, "source": rec.source, "omega_id": rec.omega_id, "attribution": rec.attribution}


def monotone_after_selection(
    ledger,
    score: float,
    selection_name: str = "",
) -> dict:
    """P-R7: keep harness/policy admitted-best non-decreasing."""
    if ledger is None:
        return {"monotone_ok": True, "reason": "no_ledger"}
    return ledger.observe(score, selection_name)


def red_queen_env_jobs(host_success: dict, base_d_t: float = 1.0) -> list[dict]:
    """P-R5: success↑ → difficulty command → evolver jobs."""
    from arsi.meta.red_queen_env import apply_to_evolver_plan, mutual_pressure_plan

    plan = mutual_pressure_plan(host_success or {}, base_d_t=base_d_t)
    return apply_to_evolver_plan(plan)


def armor_health_full(
    scores: Sequence[float],
    *,
    gdi: float = 0.0,
    metabolism_stats: Optional[dict] = None,
    monotone_history: Optional[list] = None,
    unified_ledger=None,
) -> dict:
    from arsi.harness.runtime_wiring import armor_health

    h = armor_health(
        list(scores or []),
        gdi=gdi,
        metabolism_stats=metabolism_stats,
        monotone_history=list(monotone_history or scores or []),
    )
    if unified_ledger is not None:
        try:
            h["unified_credit"] = unified_ledger.to_dict()
        except Exception:
            pass
    return h
