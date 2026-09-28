"""Runtime wiring for RSI-Armor — modules → admit / call_guard / daemon / brief.

Single entry so core/daemon do not import twenty names by hand.
"""
from __future__ import annotations

from typing import Any, Optional

from arsi.harness.conformance import ConformanceReport, conformance_report
from arsi.harness.interlock import assert_not_goal_drift, dual_sensor
from arsi.harness.overshoot import detect_overshoot
from arsi.harness.shared_calib import SharedCalibration, calibrate_shared, small_signal_policy
from arsi.harness.sahoo import capability_ceiling_hit, contractive_regime, decide_stop
from arsi.harness.everitt import everitt_guard
from arsi.harness.drawback import reliability_weighted_consensus

# process-wide conformance accumulator for call_guard
_CALL_LOG: list[tuple[str, bool]] = []


def log_call_conformance(law_id: str, used_fallback: bool) -> None:
    _CALL_LOG.append((str(law_id), bool(used_fallback)))
    if len(_CALL_LOG) > 500:
        del _CALL_LOG[: len(_CALL_LOG) - 500]


def call_conformance() -> ConformanceReport:
    return conformance_report(_CALL_LOG)


def armor_health(
    scores: list[float],
    gdi: float = 0.0,
    anchor_hash: str = "",
    prev_hash: str = "",
    *,
    metabolism_stats: Optional[dict] = None,
    monotone_history: Optional[list[float]] = None,
) -> dict:
    """Daemon health face: dual sensor + overshoot + ceiling + conformance + calib.

    Also P-R7 harness monotone + P-R8 metabolic surface when inputs provided.
    """
    from arsi.harness.metabolism import metabolic_surface, metabolism_from_stats
    from arsi.harness.monotone import harness_v_star_ge_v0

    ov = detect_overshoot(scores)
    dual = dual_sensor(scores, gdi=gdi)
    cal = calibrate_shared(scores or [0.0])
    anchored, a_reason = assert_not_goal_drift("iron_laws_sealed", anchor_hash or "h", prev_hash or anchor_hash or "h", gdi=gdi)
    try:
        from arsi.harness.gai import dual_sensor_goal_drift

        gai_alarm = dual_sensor_goal_drift(
            anchor_hash_stable=True,
            gdi=gdi,
            steps_past_best=ov.steps_past_best if ov else 0,
        )
    except Exception:
        gai_alarm = {}
    meta = (
        metabolism_from_stats(metabolism_stats)
        if metabolism_stats
        else metabolic_surface(n_new_traces=0, window_actions=1, effects=list(scores or []), cost_units=0.0)
    )
    return {
        "dual_sensor": dual.to_dict(),
        "overshoot": ov.to_dict(),
        "goal_vitals": {"gdi": gdi, "ceiling": capability_ceiling_hit([gdi] if gdi else [])},
        "conformance": call_conformance().to_dict(),
        "shared_calib": cal.to_dict(),
        "small_signal_policy": small_signal_policy(cal),
        "anchored": anchored,
        "anchor_reason": a_reason,
        "gai_dual_sensor": gai_alarm,
        "metabolism": meta.to_dict(),
        "harness_monotone": harness_v_star_ge_v0(monotone_history or list(scores or [])),
    }
    try:
        from arsi.iwm.vitals_ledger import diagnose_vitals

        out["vitals"] = diagnose_vitals({"eta": gdi, "layer1": {}}).to_dict()
    except Exception:
        out["vitals"] = {}
    try:
        from arsi.world_model.opf_discipline import opf_discipline

        z_rows = [{"live_last": s, "live_ema": s, "self_trust": gdi} for s in (scores or [])]
        if not z_rows:
            z_rows = [{"self_trust": gdi, "live_last": 0.0}]
        out["opf"] = opf_discipline(z_rows).to_dict()
    except Exception:
        out["opf"] = {}
    return out


def admit_extras(
    *,
    valid_before: float = 1.0,
    valid_after: float = 1.0,
    no_sub_before: float = 0.0,
    no_sub_after: float = 0.0,
    scores: Optional[list[float]] = None,
    gdi: float = 0.0,
    utility_rewrite: bool = False,
    value_anticipates_rewrite: bool = False,
    evaluates_with_current_utility: bool = False,
) -> tuple[bool, str, dict]:
    """Non-compensatory extras for admit(): domain guard + stop rules + Everitt."""
    from arsi.harness.accept import domain_guard

    ok, reason = domain_guard(
        valid_output_rate_before=valid_before,
        valid_output_rate_after=valid_after,
        no_submission_before=no_sub_before,
        no_submission_after=no_sub_after,
    )
    if not ok:
        return False, reason, {}
    if scores:
        stop = decide_stop(
            constraint_preservation=1.0,
            gdi=gdi,
            regression=detect_overshoot(scores).steps_past_best / 10.0,
            quality_series=scores,
        )
        if stop.stop and stop.rule in ("constraint_zero", "gdi_threshold", "regression_risk", "quality_flat"):
            return False, f"stop:{stop.rule}", {"stop": stop.to_dict()}
    if utility_rewrite:
        ev = everitt_guard(
            utility_rewrite_proposed=True,
            value_anticipates_rewrite=value_anticipates_rewrite,
            evaluates_with_current_utility=evaluates_with_current_utility,
        )
        if not ev.ok:
            return False, f"everitt:{ev.reason}", {"everitt": ev.to_dict()}
    return True, "ok", {"domain_guard": "ok"}


def brief_compress(items: list[str], max_chars: int = 4000) -> str:
    from arsi.harness.bounded_context import ContextBudget, compress_history

    return compress_history(items, ContextBudget(max_chars=max_chars)).summary
