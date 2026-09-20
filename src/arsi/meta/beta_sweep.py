"""Beta sweep — offline β grid evaluation on WorldPool (Phase D3).

Paper B.2: evaluator sweeps a fixed beta grid; ranks by
  pareto.reward = pareto.auc - lambda * parallel_penalty
ARSI default grid until official code is released: [0.2, 0.4, 0.6, 0.8, 1.0]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_BETA_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_PARALLEL_LAMBDA = 0.05


@dataclass
class BetaSweepPoint:
    beta: float
    avg_score: float = 0.0
    avg_quality: float = 0.0
    avg_probes: float = 0.0
    parallel_penalty: float = 0.0
    pareto_reward: float = 0.0

    def to_dict(self) -> dict:
        return {
            "beta": self.beta,
            "avg_score": round(self.avg_score, 4),
            "avg_quality": round(self.avg_quality, 4),
            "avg_probes": round(self.avg_probes, 4),
            "parallel_penalty": round(self.parallel_penalty, 4),
            "pareto_reward": round(self.pareto_reward, 4),
        }


@dataclass
class BetaSweepResult:
    grid: list[float] = field(default_factory=lambda: list(DEFAULT_BETA_GRID))
    points: list[BetaSweepPoint] = field(default_factory=list)
    selected_default_beta: float = 0.6
    reason: str = "bootstrap"
    world_count: int = 0

    @property
    def non_degenerate(self) -> bool:
        if len(self.points) < 2:
            return False
        scores = {round(p.avg_score, 4) for p in self.points}
        probes = {round(p.avg_probes, 2) for p in self.points}
        rewards = {round(p.pareto_reward, 4) for p in self.points}
        return len(scores) > 1 or len(probes) > 1 or len(rewards) > 1

    def to_dict(self) -> dict:
        return {
            "grid": self.grid,
            "points": [p.to_dict() for p in self.points],
            "selected_default_beta": self.selected_default_beta,
            "reason": self.reason,
            "world_count": self.world_count,
            "non_degenerate": self.non_degenerate,
        }


def _pareto_reward(point: BetaSweepPoint, lam: float) -> float:
    # auc proxy: quality with probe penalty folded into avg_score already;
    # use quality - lam * parallel_penalty as paper-style ranking signal
    return point.avg_quality - lam * point.parallel_penalty


def sweep_beta(
    pool,
    policy_factory: Callable[[float], Callable],
    grid: Optional[Sequence[float]] = None,
    parallel_lambda: float = DEFAULT_PARALLEL_LAMBDA,
    live_history: Optional[list[dict]] = None,
) -> BetaSweepResult:
    """Evaluate policy variants across the world pool for each beta in grid."""
    grid = list(grid or DEFAULT_BETA_GRID)
    result = BetaSweepResult(grid=grid)

    if pool is None or getattr(pool, "size", 0) == 0:
        result.reason = "empty_pool"
        result.selected_default_beta = 0.6
        return result

    result.world_count = int(pool.size)

    for beta in grid:
        fn = policy_factory(float(beta))
        ev = pool.evaluate_policy_across_pool(fn, policy_name=f"beta_{beta}")
        if not ev.get("available"):
            continue
        # Estimate parallel penalty from avg probes vs rounds if present
        per = ev.get("per_world") or []
        penalties = []
        for w in per:
            probes = float(w.get("probes", 0) or 0)
            rounds = float(w.get("rounds", 1) or 1)
            # serial-like when probes/rounds ≈ 1; full batching lower
            penalties.append(1.0 / max(probes / max(rounds, 1.0), 1e-6) if probes else 0.0)
        # paper: mean(effective_seq/probes); with W focus model use inverse batch factor
        # simpler stable proxy: probes / (rounds * W_est) inverted — keep small
        avg_pen = (sum(penalties) / len(penalties)) if penalties else 0.0
        # normalize: more probes per round => lower penalty
        if per:
            avg_probes = sum(float(w.get("probes", 0) or 0) for w in per) / len(per)
            avg_rounds = sum(float(w.get("rounds", 1) or 1) for w in per) / len(per)
            batch_factor = avg_probes / max(avg_rounds, 1.0)
            avg_pen = 1.0 / max(batch_factor, 1e-6)
            avg_pen = min(avg_pen, 1.0)

        point = BetaSweepPoint(
            beta=float(beta),
            avg_score=float(ev.get("avg_score", 0.0)),
            avg_quality=float(ev.get("avg_quality", 0.0)),
            avg_probes=float(ev.get("avg_probes", 0.0)),
            parallel_penalty=avg_pen,
        )
        point.pareto_reward = _pareto_reward(point, parallel_lambda)
        result.points.append(point)

    if not result.points:
        result.reason = "no_evaluable_points"
        result.selected_default_beta = 0.6
        return result

    # Cross-cycle default-beta rule (B.2), using sweep + optional live scores
    history = live_history or []
    improving = False
    if len(history) >= 2:
        s0 = float(history[-2].get("best_score", 0.0) or 0.0)
        s1 = float(history[-1].get("best_score", 0.0) or 0.0)
        improving = s1 > s0 + 1e-6
    plateau = len(history) >= 3 and not improving
    if len(history) >= 3:
        s1 = float(history[-2].get("best_score", 0.0) or 0.0)
        s2 = float(history[-1].get("best_score", 0.0) or 0.0)
        plateau = abs(s2 - s1) < 1e-3

    best_point = max(result.points, key=lambda p: (p.pareto_reward, p.avg_score))
    prev_beta = float(history[-1].get("beta", 0.6)) if history else 0.6
    same = [p for p in result.points if abs(p.beta - prev_beta) < 0.05]
    prev_point = same[0] if same else None

    # Hard rule (analysis): degenerate sweep → FREEZE beta, no thrashing
    if not result.non_degenerate:
        result.selected_default_beta = prev_beta
        result.reason = f"degenerate_freeze_beta_{prev_beta:.2f}"
        logger.info(f"β sweep degenerate → freeze beta={prev_beta}")
        result.selected_default_beta = max(0.0, min(1.0, float(result.selected_default_beta)))
        return result

    if improving and prev_point is not None:
        # keep prior unless sweep clearly shows better nearby beta
        nearby = [p for p in result.points if abs(p.beta - prev_beta) <= 0.25]
        better_nearby = [p for p in nearby if p.pareto_reward > prev_point.pareto_reward + 1e-4]
        if better_nearby:
            result.selected_default_beta = max(better_nearby, key=lambda p: p.pareto_reward).beta
            result.reason = "improving_but_sweep_nearby_better"
        else:
            result.selected_default_beta = prev_beta
            result.reason = "improving_keep_prior"
    elif plateau:
        # raise if higher beta reaches higher attainment for reasonable work
        higher = [p for p in result.points if p.beta > prev_beta + 0.05]
        good_higher = [
            p for p in higher
            if (prev_point is None or p.avg_quality >= prev_point.avg_quality)
            and (prev_point is None or p.avg_probes <= prev_point.avg_probes * 1.5 + 1)
        ]
        if good_higher:
            result.selected_default_beta = min(max(good_higher, key=lambda p: p.pareto_reward).beta, 1.0)
            result.reason = "plateau_raise_beta"
        else:
            # high beta only adds work?
            high_cost = [
                p for p in higher
                if prev_point is not None
                and p.avg_probes > prev_point.avg_probes * 1.2
                and p.avg_quality <= prev_point.avg_quality + 1e-4
            ]
            if high_cost:
                result.selected_default_beta = max(0.0, prev_beta - 0.15)
                result.reason = "plateau_high_beta_wasteful_lower"
            else:
                result.selected_default_beta = best_point.beta
                result.reason = "plateau_best_pareto"
    else:
        result.selected_default_beta = best_point.beta
        result.reason = "insufficient_live_use_best_pareto"

    result.selected_default_beta = max(0.0, min(1.0, float(result.selected_default_beta)))
    return result
