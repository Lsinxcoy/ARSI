"""T1 Conformance / fallback-rate KPI (SEVerA second/third-pass).

An FGGM that always uses the fallback is formally correct but useless.
Track fallback rate as proposal-quality × contract-tightness health signal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence


@dataclass
class ConformanceReport:
    n: int = 0
    n_fallback: int = 0
    fallback_rate: float = 0.0
    by_law: dict = field(default_factory=dict)
    advice: str = ""
    note: str = "severa_conformance_tuning"

    def to_dict(self) -> dict:
        return asdict(self)


def conformance_report(
    results: Sequence[tuple[str, bool]],
    *,
    healthy_max: float = 0.5,
    always_fallback_min: float = 0.95,
) -> ConformanceReport:
    """results: [(law_id, used_fallback), ...]"""
    n = len(results or [])
    n_fb = 0
    by: dict[str, dict] = {}
    for law, fb in results or []:
        d = by.setdefault(law, {"n": 0, "fallback": 0})
        d["n"] += 1
        if fb:
            d["fallback"] += 1
            n_fb += 1
    rate = (n_fb / n) if n else 0.0
    if n == 0:
        advice = "no_calls"
    elif rate >= always_fallback_min:
        advice = "always_fallback_formally_ok_useless_conformance_tune_or_loosen"
    elif rate > healthy_max:
        advice = "high_fallback_review_proposal_quality_and_contract_tightness"
    else:
        advice = "healthy"
    for law, d in by.items():
        d["rate"] = round(d["fallback"] / d["n"], 4) if d["n"] else 0.0
    return ConformanceReport(n=n, n_fallback=n_fb, fallback_rate=round(rate, 4), by_law=by, advice=advice)
