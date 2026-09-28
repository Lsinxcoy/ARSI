"""P-vitals: why eta / self_trust / live_layer1 are flat — accounting ledger.

Long-run data: eta≈0, self_trust=0, live_layer1=0 while holdout=0.94.
This module does NOT change scores; it records *why* so L3 can be fixed
with evidence instead of guesswork.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class VitalsLedger:
    eta: float = 0.0
    self_trust: float = 0.0
    live_layer1: float = 0.0
    holdout_layer1: float = 0.0
    iwm_total: int = 0
    iwm_success_rate: float = 0.0
    baseline_total: int = 0
    baseline_success_rate: float = 0.0
    reasons: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    note: str = "p_vitals_diagnosis_only"

    def to_dict(self) -> dict:
        return asdict(self)


def diagnose_vitals(stats: Any, iwm_report: Optional[dict] = None) -> VitalsLedger:
    """Explain flat vitals from get_stats / IWM report."""
    s = dict(stats or {})
    layer1 = s.get("layer1") or {}
    iw = iwm_report or {}
    if not iw:
        iw = s.get("iwm") or {}
        if isinstance(iw.get("iwm"), dict):
            iw = iw["iwm"]
    cal = iw.get("calibration") or {}
    v = VitalsLedger(
        eta=float(s.get("eta") or 0.0),
        self_trust=float(iw.get("self_trust") or cal.get("self_trust") or 0.0),
        live_layer1=float(layer1.get("live_accuracy") or 0.0),
        holdout_layer1=float(layer1.get("holdout_accuracy") or 0.0),
        iwm_total=int(cal.get("iwm_total") or 0),
        iwm_success_rate=float(cal.get("iwm_success_rate") or 0.0),
        baseline_total=int(cal.get("baseline_total") or 0),
        baseline_success_rate=float(cal.get("baseline_success_rate") or 0.0),
    )
    # eta
    if v.eta < 0.05:
        v.reasons.append("eta_near_zero: prediction_error_stream_empty_or_layer1_too_accurate_on_categories")
        v.fixes.append("feed explicit pred_cat≠actual_cat events into EtaTracker; do not skip evaluate()")
    # self_trust
    if v.iwm_total and v.iwm_success_rate <= 0.01:
        v.reasons.append(f"iwm_success_rate_zero_over_n={v.iwm_total}: interventions_logged_as_failures")
        v.fixes.append("IWM.observe_* must pass success=True when host outcome is success; check observe_memory(bool)")
    if v.baseline_total and v.iwm_total and v.iwm_success_rate + 0.1 < v.baseline_success_rate:
        v.reasons.append("iwm_underperforms_baseline")
    if v.self_trust < 0.05 and v.iwm_total >= 5:
        v.reasons.append("self_trust_pinned_zero_by_success_rate_and_ece")
    # live layer1
    if v.holdout_layer1 > 0.7 and v.live_layer1 < 0.05:
        v.reasons.append("layer1_holdout_high_live_zero: rules_do_not_transfer_or_live_evaluate_not_called")
        v.fixes.append("ensure SIWM.update_state always layer1.evaluate(); persist live_accuracy into stats")
    if not v.reasons:
        v.reasons.append("vitals_within_normal_bands")
    return v


def save_vitals(path: str | Path, v: VitalsLedger) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = v.to_dict()
    row["timestamp"] = datetime.now().isoformat()
    p.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
