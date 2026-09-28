"""T7 Shared calibration (third-pass F3).

δ (RRSI noise floor) · GDI threshold (SAHOO) · anchor strength (Grader) are
one family: "do not trust small signals". Calibrate once, use everywhere.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Sequence


@dataclass
class SharedCalibration:
    delta: float = 0.02
    gdi_threshold: float = 0.44
    anchor_min_checkable: int = 2
    note: str = "one_family_do_not_trust_small_signals"

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate_shared(
    baseline_scores: Sequence[float],
    drift_series: Sequence[float] | None = None,
    n_anchor_checkable: int = 2,
) -> SharedCalibration:
    """One pass: δ from H0 repeats; GDI thr from drift quantile; anchor from count."""
    xs = [float(x) for x in (baseline_scores or [])]
    if len(xs) >= 3:
        delta = max(1e-6, (max(xs) - min(xs)) / 2.0)
    else:
        delta = 0.02
    ds = [float(x) for x in (drift_series or []) if x is not None]
    if len(ds) >= 5:
        s = sorted(ds)
        gdi = round(min(0.8, max(0.2, s[int(0.9 * (len(s) - 1))])), 4)
    else:
        gdi = 0.44
    return SharedCalibration(
        delta=round(delta, 6),
        gdi_threshold=gdi,
        anchor_min_checkable=max(1, int(n_anchor_checkable)),
    )


def small_signal_policy(cal: SharedCalibration) -> dict:
    """Unified reject rules across the three systems."""
    return {
        "selection_floor": f"S_hat >= S_star - {cal.delta}",
        "vitals_alarm": f"GDI > {cal.gdi_threshold}",
        "metric_validity": f"checkable_anchors >= {cal.anchor_min_checkable}",
        "family": cal.note,
    }
