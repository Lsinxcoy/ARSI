"""GridPlan — evidence-driven width/depth for ARSI terms (Phase D2).

Dream-RSI Appendix B.2 plan_grid adapted to agent consulting:
  branch_count W = parallel focus count (dimensions/operators)
  refine_count R = refinement steps per focus

Evidence rules from live cycle manifests only (prefix-safe).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Paper bootstrap when history is empty/insufficient
DEFAULT_BOOTSTRAP_W = 2
DEFAULT_BOOTSTRAP_R = 3


@dataclass
class GridPlan:
    branch_count: int = DEFAULT_BOOTSTRAP_W
    refine_count: int = DEFAULT_BOOTSTRAP_R
    reason: str = "bootstrap"
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "branch_count": self.branch_count,
            "refine_count": self.refine_count,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass
class GridPlanningContext:
    """Prefix-safe facts for plan_grid — never current-episode outcomes."""

    history: list[dict] = field(default_factory=list)  # beta_history_row-like
    hard_max_branch_count: int = 4
    hard_max_refine_count: int = 8
    min_branch_count: int = 1
    min_refine_count: int = 1
    worker_cap: int = 4
    covered_dimensions: list[str] = field(default_factory=list)
    all_dimensions: list[str] = field(default_factory=list)
    budget_force: float = 1.0  # AdaptiveBehaviorController multiplier
    cost_budget: float = 5.0


def plan_grid(context: GridPlanningContext) -> GridPlan:
    """Deterministic grid plan from historical live manifests.

    Rules (paper B.2 → ARSI domain):
      R1 width-useful / depth-stall     → +W, hold/-R
      R2 late gains on few directions   → +R, hold/-W
      R3 plateau + uncovered dim classes → +W
      R4 hard fails / redundant dims     → conservative -W -R
      else insufficient/conflict         → bootstrap
    """
    hist = [h for h in (context.history or []) if isinstance(h, dict)]
    max_w = max(context.min_branch_count, min(context.hard_max_branch_count, context.worker_cap))
    max_r = max(context.min_refine_count, context.hard_max_refine_count)
    min_w = context.min_branch_count
    min_r = context.min_refine_count

    def clamp(w: int, r: int) -> tuple[int, int]:
        return max(min_w, min(max_w, w)), max(min_r, min(max_r, r))

    if len(hist) < 2:
        w, r = clamp(DEFAULT_BOOTSTRAP_W, DEFAULT_BOOTSTRAP_R)
        return GridPlan(
            branch_count=w,
            refine_count=r,
            reason="bootstrap_insufficient_evidence",
            evidence={"history_len": len(hist)},
        )

    recent = hist[-5:]
    scores = [float(h.get("best_score", 0.0) or 0.0) for h in recent]
    widths = [int(h.get("effective_w", h.get("planned_w", DEFAULT_BOOTSTRAP_W)) or DEFAULT_BOOTSTRAP_W) for h in recent]
    depths = [int(h.get("effective_r", h.get("planned_r", DEFAULT_BOOTSTRAP_R)) or DEFAULT_BOOTSTRAP_R) for h in recent]
    probes = [int(h.get("probe_work", 0) or 0) for h in recent]
    best_w, best_r = widths[-1], depths[-1]

    # Score deltas
    deltas = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
    last_delta = deltas[-1] if deltas else 0.0
    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

    # Depth-stall proxy: deeper terms without better scores, OR uniform high depth with weak gains
    deep_pairs = [(depths[i], deltas[i - 1]) for i in range(1, len(depths)) if i - 1 < len(deltas)]
    deep_stall = False
    late_gain = False
    if deep_pairs:
        deep_only = [d for dep, d in deep_pairs if dep >= max(depths) - 0]
        shallow_only = [d for dep, d in deep_pairs if dep < max(depths)]
        if deep_only and shallow_only:
            if sum(deep_only) / len(deep_only) <= sum(shallow_only) / len(shallow_only) + 1e-9:
                deep_stall = True
        # late gain: most recent delta strong while previous were weak
        if last_delta > max(avg_delta * 1.5, 0.02) and depths[-1] >= 3:
            late_gain = True
        if last_delta > 0.05 and avg_delta > 0 and depths[-1] >= max(depths):
            late_gain = True

    # Uniform high depth with only weak gains → depth is saturated (R1)
    if depths and min(depths) >= 3 and max(depths) == min(depths) and avg_delta < 0.05 and last_delta < 0.05:
        deep_stall = True

    # True stagnation only — mild positive deltas are NOT R4
    stagnation = (
        len(deltas) >= 2
        and abs(deltas[-1]) < 0.005
        and abs(deltas[-2]) < 0.005
        and last_delta <= 0.005
    )
    hardish = last_delta < -0.05

    uncovered = []
    if context.all_dimensions:
        covered = set(context.covered_dimensions or [])
        uncovered = [d for d in context.all_dimensions if d not in covered]

    # Order: hard fail > R1 width/depth trade > R2 deepen > R3/R4 > hold
    if hardish:
        w, r = clamp(best_w - 1, best_r - 1)
        reason = "R4_hard_or_regressive"
    elif deep_stall and not late_gain:
        w, r = clamp(best_w + 1, max(min_r, best_r - 1))
        reason = "R1_width_useful_depth_stall"
    elif late_gain:
        w, r = clamp(max(min_w, best_w - 1) if best_w > max_w - 1 else best_w, best_r + 1)
        reason = "R2_late_gain_deepen"
    elif stagnation and not uncovered:
        w, r = clamp(max(min_w, best_w - 1), max(min_r, best_r - 1))
        reason = "R4_redundant_plateau"
    elif stagnation and uncovered:
        w, r = clamp(best_w + 1, best_r)
        reason = "R3_uncovered_dimensions"
    else:
        w, r = clamp(best_w, best_r)
        reason = "hold_evidence_neutral"

    # Adaptive force: performance rising → conserve R; plateau → expand R
    force = float(context.budget_force or 1.0)
    if force < 0.9:
        r = max(min_r, int(round(r * force)))
        reason = reason + ";adaptive_conserve"
    elif force > 1.1:
        r = min(max_r, int(round(r * force)))
        reason = reason + ";adaptive_expand"

    # Budget sketch: cost ≈ W * R * unit_cost (unit=1.0 default)
    unit = 1.0
    est = w * r * unit
    if est > context.cost_budget > 0:
        # shrink R first then W
        while est > context.cost_budget and r > min_r:
            r -= 1
            est = w * r * unit
        while est > context.cost_budget and w > min_w:
            w -= 1
            est = w * r * unit
        reason = reason + ";budget_clamped"

    return GridPlan(
        branch_count=w,
        refine_count=r,
        reason=reason,
        evidence={
            "history_len": len(hist),
            "last_best": scores[-1] if scores else 0.0,
            "last_delta": round(last_delta, 4),
            "avg_delta": round(avg_delta, 4),
            "uncovered": uncovered[:6],
            "force": force,
            "est_cost": round(est, 3),
        },
    )
