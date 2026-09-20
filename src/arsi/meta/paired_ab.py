"""Paired A/B selection with effect-size gate — SYNTHEX P1/S6.

SelectionGate lesson: absolute scores on degenerate pools are noise.
ARSI adopts policies only when paired per-world deltas show a real effect.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

NEGLIGIBLE_COHEN_D = 0.2
DEFAULT_MIN_SAMPLES = 5
DEFAULT_MEAN_MARGIN = 0.0


@dataclass
class PairedComparison:
    base_name: str = ""
    cand_name: str = ""
    n_worlds: int = 0
    base_scores: list[float] = field(default_factory=list)
    cand_scores: list[float] = field(default_factory=list)
    deltas: list[float] = field(default_factory=list)
    mean_delta: float = 0.0
    std_delta: float = 0.0
    cohen_d: float = 0.0
    promote: bool = False
    hold: bool = True
    reason: str = ""
    margin: float = DEFAULT_MEAN_MARGIN
    min_d: float = NEGLIGIBLE_COHEN_D
    min_samples: int = DEFAULT_MIN_SAMPLES

    @property
    def negligible(self) -> bool:
        return abs(self.cohen_d) < self.min_d

    def to_dict(self) -> dict:
        return asdict(self)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def cohen_d_paired(deltas: list[float]) -> float:
    """d_z = mean(delta) / sd(delta)."""
    if len(deltas) < 2:
        return 0.0
    sd = _std(deltas)
    if sd < 1e-12:
        # all deltas identical: huge effect only if nonzero mean
        m = _mean(deltas)
        return 0.0 if abs(m) < 1e-12 else (10.0 if m > 0 else -10.0)
    return _mean(deltas) / sd


def compare_paired(
    base_scores: list[float],
    cand_scores: list[float],
    base_name: str = "base",
    cand_name: str = "cand",
    margin: float = DEFAULT_MEAN_MARGIN,
    min_d: float = NEGLIGIBLE_COHEN_D,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_mean_gain: float = 0.01,
) -> PairedComparison:
    """Pair by world index; promote only with sufficient n + real mean gain + effect size."""
    n = min(len(base_scores), len(cand_scores))
    base = [float(x) for x in base_scores[:n]]
    cand = [float(x) for x in cand_scores[:n]]
    deltas = [c - b for c, b in zip(cand, base)]
    mean_d = _mean(deltas)
    std_d = _std(deltas)
    d = cohen_d_paired(deltas)

    result = PairedComparison(
        base_name=base_name,
        cand_name=cand_name,
        n_worlds=n,
        base_scores=base,
        cand_scores=cand,
        deltas=deltas,
        mean_delta=round(mean_d, 6),
        std_delta=round(std_d, 6),
        cohen_d=round(d, 4),
        margin=margin,
        min_d=min_d,
        min_samples=min_samples,
    )

    if n < min_samples:
        result.promote = False
        result.hold = True
        result.reason = f"hold_insufficient_samples_n={n}<{min_samples}"
        return result
    # Tiny mean gains are noise even if d looks large (low variance)
    if abs(mean_d) < min_mean_gain:
        result.promote = False
        result.hold = True
        result.reason = f"hold_tiny_mean_delta={result.mean_delta}"
        return result
    if result.negligible:
        result.promote = False
        result.hold = True
        result.reason = f"hold_negligible_effect_d={result.cohen_d}<{min_d}"
        return result
    if mean_d > max(margin, min_mean_gain) and d >= min_d:
        result.promote = True
        result.hold = False
        result.reason = f"promote_mean_delta={result.mean_delta}_d={result.cohen_d}"
        return result
    if mean_d < -max(margin, min_mean_gain) and d <= -min_d:
        result.promote = False
        result.hold = False
        result.reason = f"reject_cand_worse_mean_delta={result.mean_delta}_d={result.cohen_d}"
        return result
    result.promote = False
    result.hold = True
    result.reason = f"hold_no_clear_effect_mean_delta={result.mean_delta}_d={result.cohen_d}"
    return result


def select_policy_paired(
    pool,
    current_fn,
    candidate_fns: list,
    current_name: str = "current",
    candidate_names: Optional[list[str]] = None,
    margin: float = DEFAULT_MEAN_MARGIN,
    min_d: float = NEGLIGIBLE_COHEN_D,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict:
    """S6: champion/challenger on per-world paired scores.

    Current policy always remains candidate; challenger must beat it with effect size.
    """
    names = candidate_names or [f"cand_{i+1}" for i in range(len(candidate_fns))]
    base_eval = pool.evaluate_policy_across_pool(current_fn, policy_name=current_name)
    base_scores = [
        float(w.get("score", 0.0))
        for w in (base_eval.get("per_world") or [])
    ]

    comparisons = []
    best_name = current_name
    best_fn = current_fn
    best_eval = base_eval
    best_cmp = None

    for name, fn in zip(names, candidate_fns):
        ev = pool.evaluate_policy_across_pool(fn, policy_name=name)
        cand_scores = [float(w.get("score", 0.0)) for w in (ev.get("per_world") or [])]
        cmp = compare_paired(
            base_scores,
            cand_scores,
            base_name=current_name,
            cand_name=name,
            margin=margin,
            min_d=min_d,
            min_samples=min_samples,
        )
        comparisons.append(cmp.to_dict())
        if cmp.promote:
            # pick challenger with largest mean_delta among promotes
            if best_cmp is None or cmp.mean_delta > best_cmp.mean_delta:
                best_name = name
                best_fn = fn
                best_eval = ev
                best_cmp = cmp

    return {
        "best_name": best_name,
        "best_fn": best_fn,
        "best_eval": best_eval,
        "current_name": current_name,
        "current_eval": base_eval,
        "paired_comparisons": comparisons,
        "promoted": best_name != current_name,
        "gate": {
            "min_d": min_d,
            "margin": margin,
            "min_samples": min_samples,
            "rule": "paired_delta_mean+cohen_d; negligible→hold",
        },
        "monotone_ok": True,  # current always eligible; promote only on evidence
        "all": [
            {"name": c["cand_name"], "mean_delta": c["mean_delta"], "cohen_d": c["cohen_d"],
             "promote": c["promote"], "reason": c["reason"],
             "avg_score": _mean(c["cand_scores"]),
             "world_count": c["n_worlds"]}
            for c in comparisons
        ] + [{"name": current_name, "mean_delta": 0.0, "cohen_d": 0.0,
              "promote": best_name == current_name,
              "reason": "champion",
              "avg_score": float(base_eval.get("avg_score", 0.0) or 0.0),
              "world_count": len(base_scores)}],
    }
