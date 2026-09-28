"""SAHOO-style Goal Drift Index + stop rules (arXiv:2603.06333).

GDI = w_sem·Δ_sem + w_lex·Δ_lex + w_str·Δ_str + w_dist·Δ_dist
Paper calibrated weights: 0.38 / 0.12 / 0.21 / 0.29 — ARSI must recalibrate.

Stop priority (first hit wins):
  1. constraint_preservation == 0   (absolute)
  2. GDI > threshold
  3. regression_risk > threshold
  4. quality flat (ΔQ < eps for n consecutive)
  5. max cycles
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

DEFAULT_WEIGHTS = {
    "semantic": 0.38,
    "lexical": 0.12,
    "structural": 0.21,
    "distributional": 0.29,
}

# Paper-calibrated; ARSI must recalibrate on local traces (App.B).
WEIGHTS_STATUS = {
    "source": "paper_2603.06333",
    "must_recalibrate": True,
    "gdi_threshold_paper": 0.44,
}


def car_frontier(car_series: Sequence[float]) -> dict:
    """Capability–alignment frontier over cycles (SAHOO mapping).

    Paper: CAR high early, settles 0.6–0.7; later cycles cost more alignment.
    """
    xs = [float(x) for x in (car_series or [])]
    if not xs:
        return {"n": 0, "note": "empty_car_series"}
    early = xs[: max(1, len(xs) // 3)]
    late = xs[-max(1, len(xs) // 3) :]
    e = sum(early) / len(early)
    l = sum(late) / len(late)
    return {
        "n": len(xs),
        "early_mean": round(e, 4),
        "late_mean": round(l, 4),
        "trend": "alignment_cost_rising" if l < e - 0.05 else ("improving" if l > e + 0.05 else "flat"),
        "weights_status": dict(WEIGHTS_STATUS),
        "note": "sahoo_capability_alignment_frontier",
    }


@dataclass
class DriftReport:
    gdi: float
    components: dict
    over_threshold: bool
    note: str = "sahoo_gdi_rulebased"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StopVerdict:
    stop: bool
    rule: str
    priority: int
    detail: dict = field(default_factory=dict)


def _tv_bernoulli(p: float, q: float) -> float:
    p = min(1.0, max(0.0, float(p)))
    q = min(1.0, max(0.0, float(q)))
    return abs(p - q)


def lexical_drift(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    """L1/2 of unigram distributions (proxy for lexical drift)."""
    if not tokens_a or not tokens_b:
        return 0.0
    from collections import Counter

    ca, cb = Counter(tokens_a), Counter(tokens_b)
    na, nb = sum(ca.values()), sum(cb.values())
    keys = set(ca) | set(cb)
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in keys)


def structural_drift(shape_a: dict, shape_b: dict) -> float:
    """Normalized L1 on structural feature vectors (lens, sections, n_edits…)."""
    keys = set(shape_a or {}) | set(shape_b or {})
    if not keys:
        return 0.0
    s = 0.0
    for k in keys:
        va = float((shape_a or {}).get(k) or 0.0)
        vb = float((shape_b or {}).get(k) or 0.0)
        s += abs(va - vb)
    return min(1.0, s / max(1, len(keys)))


def distributional_drift(h_a: Sequence[float], h_b: Sequence[float]) -> float:
    """TV distance between histograms (normalized)."""
    n = min(len(h_a), len(h_b))
    if n == 0:
        return 0.0
    sa = sum(h_a[:n]) or 1.0
    sb = sum(h_b[:n]) or 1.0
    return 0.5 * sum(abs(h_a[i] / sa - h_b[i] / sb) for i in range(n))


def semantic_drift_proxy(text_a: str, text_b: str, ngram: int = 3) -> float:
    """Char-ngram Jaccard distance — semantic proxy without embeddings."""
    def grams(s: str) -> set[str]:
        s = (s or "").lower()
        return {s[i : i + ngram] for i in range(max(0, len(s) - ngram + 1))}

    ga, gb = grams(text_a), grams(text_b)
    if not ga and not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb) or 1
    return 1.0 - inter / union


def goal_drift_index(
    *,
    text_a: str = "",
    text_b: str = "",
    tokens_a: Sequence[str] = (),
    tokens_b: Sequence[str] = (),
    shape_a: Optional[dict] = None,
    shape_b: Optional[dict] = None,
    hist_a: Optional[Sequence[float]] = None,
    hist_b: Optional[Sequence[float]] = None,
    weights: Optional[dict] = None,
    threshold: float = 0.44,
) -> DriftReport:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    comps = {
        "semantic": round(semantic_drift_proxy(text_a, text_b), 4),
        "lexical": round(lexical_drift(tokens_a or text_a.split(), tokens_b or text_b.split()), 4),
        "structural": round(structural_drift(shape_a or {}, shape_b or {}), 4),
        "distributional": round(distributional_drift(hist_a or [], hist_b or []), 4),
    }
    gdi = (
        w["semantic"] * comps["semantic"]
        + w["lexical"] * comps["lexical"]
        + w["structural"] * comps["structural"]
        + w["distributional"] * comps["distributional"]
    )
    return DriftReport(
        gdi=round(gdi, 4),
        components=comps,
        over_threshold=gdi > threshold,
        note="weights_must_recalibrate_on_arsi",
    )


def regression_risk(
    quality_series: Sequence[float],
    volatility: Optional[float] = None,
) -> float:
    """SAHOO App.D — volatility / trend / normalized gap → risk in [0,1]."""
    xs = [float(x) for x in quality_series or []]
    if len(xs) < 3:
        return 0.0
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    vol = float(volatility) if volatility is not None else math.sqrt(var)
    # slope via least squares on index
    tbar = (n - 1) / 2
    sxy = sum((i - tbar) * (xs[i] - mu) for i in range(n))
    sxx = sum((i - tbar) ** 2 for i in range(n)) or 1.0
    slope = sxy / sxx
    gap = (xs[-1] - max(xs)) / (abs(max(xs)) + 1e-6)
    vol_n = abs(gap) / (vol + 1e-6)
    # combine: down-slope + high vol + large drop from peak
    risk = 0.4 * min(1.0, max(0.0, -slope * n)) + 0.3 * min(1.0, vol_n / 3.0) + 0.3 * min(1.0, abs(min(0.0, gap)) * 3)
    return round(max(0.0, min(1.0, risk)), 4)


def capability_ceiling_hit(
    car_series,
    threshold: float = 0.3,
    n: int = 3,
) -> bool:
    xs = [float(x) for x in (car_series or [])]
    if len(xs) < n:
        return False
    return (sum(xs[-n:]) / len(xs[-n:])) < threshold


def contractive_regime(quality_gain_mean: float, drift_response: float, eps: float = 1e-9) -> dict:
    mu = float(quality_gain_mean)
    ld = float(drift_response) / max(abs(mu), eps) if mu else float("inf")
    bounded = ld < 1.0
    asymptote = (ld * mu / (1.0 - ld)) if bounded and mu else None
    return {
        "L_delta": round(ld, 6),
        "contractive": bounded,
        "drift_asymptote": round(asymptote, 6) if asymptote is not None else None,
        "note": "contractive_iff_L_lt_1",
    }


def capability_alignment_ratio(quality_gain: float, drift: float, eps: float = 1e-3) -> float:
    """CAR — gain per unit alignment cost (SAHOO App.E)."""
    d = max(float(drift), eps)
    return round(float(quality_gain) / d, 4)


def decide_stop(
    *,
    constraint_preservation: float,
    gdi: float,
    regression: float,
    quality_series: Sequence[float],
    gdi_threshold: float = 0.44,
    regression_threshold: float = 0.8,
    flat_eps: float = 0.01,
    flat_n: int = 3,
    cycle: int = 0,
    max_cycles: int = 20,
) -> StopVerdict:
    """Non-compensatory stop rules — constraint=0 is absolute."""
    if float(constraint_preservation) <= 0.0:
        return StopVerdict(True, "constraint_zero", 1, {"cps": constraint_preservation})
    if float(gdi) > gdi_threshold:
        return StopVerdict(True, "gdi_threshold", 2, {"gdi": gdi, "thr": gdi_threshold})
    if float(regression) > regression_threshold:
        return StopVerdict(True, "regression_risk", 3, {"risk": regression, "thr": regression_threshold})
    xs = [float(x) for x in quality_series or []]
    if len(xs) >= flat_n:
        recent = xs[-flat_n:]
        if max(recent) - min(recent) < flat_eps:
            return StopVerdict(True, "quality_flat", 4, {"recent": recent})
    if cycle >= max_cycles:
        return StopVerdict(True, "max_cycles", 5, {"cycle": cycle})
    return StopVerdict(False, "continue", 99, {"cycle": cycle})
