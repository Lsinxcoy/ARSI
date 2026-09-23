"""Difficulty × Flow dual gate (ODEWorld × Environment Evolution, P2-9).

Rule: rising environment difficulty (D_T) is NOT promote evidence by itself.
A policy only promotes when:
  1. paired effect-size gate already passed, AND
  2. dyn flow shows positive progress toward z_subgoal (or D_T is not rising)

Also: same-lineage-generation pairing — never mix gens in one A/B table
when evolution has already spawned children (paper EL / held-out spirit).

Tracks stay split: this gate never rewrites live_capability or pool scores.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence


def flow_progress_score(
    z_now: dict,
    z_subgoal: dict,
    v_hat: dict,
    keys: Optional[Sequence[str]] = None,
    eps: float = 1e-6,
) -> dict:
    """How much dyn state is moving toward subgoal.

    progress_k = sign(goal_k - now_k) * v_hat_k   (per key)
    score = mean of positive-aligned progress, normalized by gap size.
    """
    z_now = z_now or {}
    z_subgoal = z_subgoal or {}
    v_hat = v_hat or {}
    use_keys = list(keys) if keys else [k for k in z_subgoal.keys() if k in z_now]
    if not use_keys:
        return {"score": 0.0, "n_keys": 0, "aligned": 0, "anti": 0, "note": "empty_goal_or_z"}

    aligned = 0
    anti = 0
    scores = []
    for k in use_keys:
        goal = float(z_subgoal.get(k, 0.0) or 0.0)
        now = float(z_now.get(k, 0.0) or 0.0)
        v = float(v_hat.get(k, 0.0) or 0.0)
        gap = goal - now
        if abs(gap) < eps:
            scores.append(1.0)  # already there
            aligned += 1
            continue
        # want v to point along gap
        proj = (gap * v) / (abs(gap) + eps)
        if proj > 0:
            aligned += 1
        elif proj < 0:
            anti += 1
        # normalize: clip projection relative to |gap|
        scores.append(max(-1.0, min(1.0, proj / (abs(gap) + eps))))

    mean_s = sum(scores) / len(scores)
    return {
        "score": round(mean_s, 6),
        "n_keys": len(use_keys),
        "aligned": aligned,
        "anti": anti,
        "positive_velocity": mean_s > 0.05,
        "note": "z_progress_along_subgoal",
    }


def d_t_is_rising(
    d_t_history: Sequence[float],
    window: int = 3,
    eps: float = 0.05,
) -> tuple[bool, str]:
    """True if recent D_T mean/spread trend is harder than prior window."""
    xs = [float(x) for x in d_t_history if x is not None]
    if len(xs) < window + 1:
        return False, f"insufficient_d_t_history n={len(xs)}"
    recent = xs[-window:]
    prior = xs[-(2 * window) : -window]
    rec = sum(recent) / len(recent)
    pri = sum(prior) / len(prior)
    if rec > pri + eps:
        return True, f"d_t_rising recent={rec:.4f}>prior={pri:.4f}+eps"
    return False, f"d_t_stable_or_down recent={rec:.4f} prior={pri:.4f}"


def difficulty_flow_gate(
    d_t_history: Sequence[float],
    flow_guidance: Optional[dict] = None,
    min_progress: float = 0.0,
    require_progress_when_harder: bool = True,
) -> dict:
    """Block promote when env is harder but strategy is not flowing toward z_goal.

    Passes (allow_promote=True) when:
      - D_T not rising, OR
      - flow progress score > min_progress (positive velocity toward subgoal)
      - no flow evidence → hold (never invent progress)
    """
    fg = flow_guidance or {}
    z_now = fg.get("z_now") or {}
    z_subgoal = fg.get("z_subgoal") or {}
    v_hat = fg.get("v_hat") or {}
    prog = flow_progress_score(z_now, z_subgoal, v_hat)
    rising, rise_reason = d_t_is_rising(d_t_history)

    has_flow = bool(z_subgoal) and bool(v_hat)
    if not require_progress_when_harder:
        return {
            "allow_promote": True,
            "progress": prog,
            "d_t_rising": rising,
            "d_t_reason": rise_reason,
            "rule": "gate_disabled",
        }

    if not rising:
        allow = True
        reason = f"pass_d_t_not_rising;{rise_reason}"
    elif not has_flow:
        allow = False
        reason = "hold_harder_env_but_no_flow_evidence"
    elif prog.get("score", 0.0) > min_progress and prog.get("positive_velocity"):
        allow = True
        reason = f"pass_harder_env_with_positive_z_progress score={prog.get('score')}"
    else:
        allow = False
        reason = (
            f"block_harder_env_without_z_progress "
            f"score={prog.get('score')} aligned={prog.get('aligned')}/{prog.get('n_keys')} anti={prog.get('anti')}"
        )

    return {
        "allow_promote": bool(allow),
        "reason": reason,
        "progress": prog,
        "d_t_rising": rising,
        "d_t_reason": rise_reason,
        "min_progress": min_progress,
        "rule": "difficulty_x_flow_dual_gate",
        "note": "D_T↑ alone is not promote; need z progress or stable difficulty",
    }


def extract_d_t_history(pool, limit: int = 12) -> list[float]:
    """d_t_dyn preferred (static-decoupled), else full d_t, newest last."""
    out: list[float] = []
    for w in list(getattr(pool, "worlds", []) or [])[-limit:]:
        dyn = getattr(w, "env_difficulty_dyn", None) or {}
        full = getattr(w, "env_difficulty", None) or {}
        v = dyn.get("d_t", full.get("d_t"))
        if v is not None:
            out.append(float(v))
    return out


def same_generation_pairs(
    per_world: Sequence[dict],
    generation: Optional[int] = None,
) -> list[dict]:
    """Filter per-world score rows to one lineage generation (or dominant gen)."""
    rows = list(per_world or [])
    gens = []
    for r in rows:
        lin = r.get("lineage") or {}
        g = lin.get("generation", r.get("generation"))
        if g is not None:
            gens.append(int(g))
    if not gens:
        return rows
    if generation is None:
        # majority / max generation among rows (active EL target)
        generation = max(set(gens), key=gens.count)
    out = []
    for r in rows:
        lin = r.get("lineage") or {}
        g = lin.get("generation", r.get("generation"))
        if g is None or int(g) == int(generation):
            out.append(r)
    return out


def apply_dual_gate_to_selection(
    selection: dict,
    pool,
    flow_guidance: Optional[dict] = None,
    arsi=None,
) -> dict:
    """Post-process paired selection: revoke promote if dual gate fails.

    Does not change scores — only selection decision + reasons.
    """
    if flow_guidance is None and arsi is not None:
        cf = getattr(arsi, "capability_flow", None)
        if cf is not None:
            try:
                flow_guidance = cf.flow_guidance()
            except Exception:
                flow_guidance = {}
    d_hist = extract_d_t_history(pool)
    gate = difficulty_flow_gate(d_hist, flow_guidance)
    selection = dict(selection or {})
    selection["difficulty_flow_gate"] = gate

    promoted = bool(selection.get("promoted") or (
        selection.get("best_name")
        and selection.get("current_name")
        and selection.get("best_name") != selection.get("current_name")
    ))
    if promoted and not gate.get("allow_promote", True):
        # revoke: keep current champion
        current_name = selection.get("current_name") or selection.get("best_name")
        selection["best_name"] = current_name
        selection["promoted"] = False
        selection["gate_revoked"] = True
        selection["revoke_reason"] = gate.get("reason", "dual_gate")
        # mark paired comparisons
        comps = selection.get("paired_comparisons") or []
        for c in comps:
            if c.get("promote"):
                c["promote"] = False
                c["hold"] = True
                c["reason"] = (c.get("reason") or "") + f";dual_gate:{gate.get('reason')}"
        for row in selection.get("all") or []:
            if row.get("promote") and row.get("name") != current_name:
                row["promote"] = False
                row["reason"] = (row.get("reason") or "") + ";dual_gate_revoked"
    elif promoted:
        selection["gate_note"] = gate.get("reason")
    return selection
