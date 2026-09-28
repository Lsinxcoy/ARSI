"""I6 live acceptance gate — claim introspective world model v1 only with live evidence.

Discipline (IWM §I6 / CTM honesty):
- Q1–Q3 any two fail → skeleton (never v1)
- live_v1 additionally requires **thick live facts** from daemon-fed data
  (pool, organ samples, flow updates, calibration, paired/eval evidence)
- Seeded-runner probes only earn `introspective_v1_candidate`
- Every True claim must bind a VerifiedClaim (pytest/exit + timestamp)

Does not invent scores. Missing evidence → lower claim, never upgrade.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from arsi.foundation.verified import VerifiedClaim, unverified

__all__ = [
    "I6LiveFacts",
    "I6LiveResult",
    "DEFAULT_THRESHOLDS",
    "evaluate_i6_live",
    "facts_from_health_rows",
    "facts_from_checkpoint",
]

# Thick-data floors for the word "live" (not candidate)
DEFAULT_THRESHOLDS = {
    "min_world_pool": 5,
    "min_trace_count": 5000,
    "min_manifest_cycles": 20,
    "min_n_v_updates": 8,
    "min_layer1_holdout": 0.75,
    "min_organ_measured_frac": 0.4,
    "min_calibration_samples": 5,
    "min_health_rows": 10,
    "min_host_success_rate": 0.5,
}


@dataclass
class I6LiveFacts:
    """Aggregated live measurements — caller supplies evidence, never guessed."""

    q_gate_ok: bool = False
    q_failed: list = field(default_factory=list)
    q_pass: dict = field(default_factory=dict)
    world_pool_size: int = 0
    trace_count: int = 0
    experience_count: int = 0
    manifest_cycles: int = 0
    n_v_updates: int = 0
    layer1_holdout: float = 0.0
    organ_trust: dict = field(default_factory=dict)
    organ_measured_frac: float = 0.0
    calibration_n: int = 0
    self_trust: float = 0.0
    degrade_to_baseline: bool = False
    paired_promote: bool = False
    live_regression: bool = False
    health_rows: int = 0
    host_success_rate: float = 0.0
    suite_verified: Optional[dict] = None
    source: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class I6LiveResult:
    claim: str = "introspective_skeleton"
    live_v1: bool = False
    gates: dict = field(default_factory=dict)
    failed: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    verified: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)


def _check(facts: I6LiveFacts, thr: dict) -> tuple[dict, list[str]]:
    """Return (per-gate ok map, fail list). q_not_run ≠ skeleton."""
    q = dict(facts.q_pass or {})
    q_run = bool(q)
    # Q1–Q3 hard: any two fail → skeleton; if Q not run, we cannot claim skeleton evidence
    hard = [k for k in ("Q1_organs", "Q2_frontier", "Q3_loop") if not q.get(k, False)]
    q_ok = q_run and len(hard) < 2
    q_two_fail = q_run and len(hard) >= 2

    gates = {
        "q1_q3_introspective": q_ok,
        "q_evaluated": q_run,
        "pool_thick": facts.world_pool_size >= thr["min_world_pool"],
        "traces_thick": facts.trace_count >= thr["min_trace_count"],
        "manifest_thick": facts.manifest_cycles >= thr["min_manifest_cycles"],
        "flow_thick": facts.n_v_updates >= thr["min_n_v_updates"],
        "layer1_holdout": facts.layer1_holdout >= thr["min_layer1_holdout"],
        "organs_measured": facts.organ_measured_frac >= thr["min_organ_measured_frac"],
        "calibration_samples": facts.calibration_n >= thr["min_calibration_samples"],
        "not_degraded": (not facts.degrade_to_baseline) and facts.self_trust > 0.35,
        "health_history": facts.health_rows >= thr["min_health_rows"],
        "host_loop_healthy": facts.host_success_rate >= thr["min_host_success_rate"],
        "no_live_regression": (not facts.live_regression),
        "suite_verified": bool((facts.suite_verified or {}).get("verified")),
        "paired_or_clean_eval": bool(facts.paired_promote or not facts.live_regression),
    }
    soft_fail = [k for k, ok in gates.items() if not ok]
    return gates, soft_fail, q_two_fail, q_run


def evaluate_i6_live(
    facts: I6LiveFacts,
    thresholds: Optional[dict] = None,
    suite_claim: Optional[VerifiedClaim] = None,
) -> I6LiveResult:
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    gates, failed, q_two_fail, q_run = _check(facts, thr)

    # Claim ladder — never skip rungs
    if q_two_fail:
        claim = "introspective_skeleton"
        live_v1 = False
    else:
        live_gates = [
            "q1_q3_introspective",
            "q_evaluated",
            "pool_thick",
            "traces_thick",
            "manifest_thick",
            "flow_thick",
            "layer1_holdout",
            "organs_measured",
            "calibration_samples",
            "not_degraded",
            "health_history",
            "host_loop_healthy",
            "no_live_regression",
            "suite_verified",
            "paired_or_clean_eval",
        ]
        if all(gates.get(g, False) for g in live_gates):
            claim = "introspective_v1_live"
            live_v1 = True
        else:
            claim = "introspective_v1_candidate"
            live_v1 = False

    if suite_claim is not None:
        verified = suite_claim.to_dict()
    elif facts.suite_verified:
        verified = dict(facts.suite_verified)
    else:
        verified = unverified(claim, "no_suite_evidence").to_dict()

    # hard rule: live_v1 only with verified=True
    if live_v1 and not verified.get("verified"):
        claim = "introspective_v1_candidate"
        live_v1 = False
        failed = list(failed) + ["suite_not_verified"]
        verified = unverified(claim, "suite_not_verified_downgrade").to_dict()

    return I6LiveResult(
        claim=claim,
        live_v1=live_v1,
        gates=gates,
        failed=sorted(set(list(facts.q_failed or []) + failed)),
        facts=facts.to_dict(),
        verified=verified,
        thresholds=thr,
    )


def facts_from_health_rows(rows: list[dict]) -> I6LiveFacts:
    """Derive live thickness facts from arsi_health.jsonl tail rows."""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    if not rows:
        return I6LiveFacts(health_rows=0, note="no_health_rows")
    last = rows[-1]
    cf = last.get("capability_flow") or {}
    layer1 = last.get("layer1") or {}
    iw = last.get("iwm") or {}
    if isinstance(iw, dict) and isinstance(iw.get("iwm"), dict):
        iw = iw["iwm"]
    iw = iw if isinstance(iw, dict) else {}
    organs = iw.get("organ_trust") or {}
    if isinstance(organs, dict) and organs:
        measured = sum(1 for v in organs.values() if isinstance(v, (int, float)) and v > 0.05)
        frac = measured / max(1, len(organs))
    else:
        frac = 0.0
    cal = iw.get("calibration") or (iw.get("q_gate") or {}).get("scores", {}).get("Q6_calibrate") or {}
    if not isinstance(cal, dict):
        cal = {}
    hl = last.get("host_loop") or {}
    hosts = hl.get("hosts") or {}
    host_loop_rate = 0.0
    if isinstance(hosts, dict) and hosts:
        oks = [1 for h in hosts.values() if isinstance(h, dict) and h.get("success")]
        host_loop_rate = len(oks) / len(hosts)
    ma = last.get("multi_agent") or {}
    agents = ma.get("agents") or {}
    rates = []
    for a in agents.values():
        if isinstance(a, dict) and a.get("tasks_completed", 0) > 0:
            rates.append(float(a.get("success_rate") or 0.0))
    ma_rate = (sum(rates) / len(rates)) if rates else 0.0
    # evidence of healthy hosts = best measured arm (do not let empty MA zero out host_loop)
    host_rate = max(host_loop_rate, ma_rate)

    return I6LiveFacts(
        world_pool_size=int(last.get("world_pool_size") or 0),
        trace_count=int(last.get("trace_count") or 0),
        experience_count=int(last.get("experience_count") or 0),
        manifest_cycles=int(last.get("manifest_cycles") or 0),
        n_v_updates=int(cf.get("n_v_updates") or 0),
        layer1_holdout=float(layer1.get("holdout_accuracy") or iw.get("layer1_holdout") or 0.0),
        organ_trust=dict(organs) if isinstance(organs, dict) else {},
        organ_measured_frac=round(frac, 4),
        calibration_n=int(cal.get("n") or 0),
        self_trust=float(iw.get("self_trust") or 0.0),
        degrade_to_baseline=bool(iw.get("degrade_to_baseline")),
        health_rows=len(rows),
        host_success_rate=round(host_rate, 4),
        source="arsi_health.jsonl",
    )


def facts_from_checkpoint(ckpt: dict) -> I6LiveFacts:
    """Facts from archive/arsi_checkpoint.json (daemon snapshot)."""
    ckpt = ckpt or {}
    stats = ckpt.get("stats") or {}
    iwm = stats.get("iwm") or ckpt.get("iwm") or {}
    if isinstance(iwm, dict) and isinstance(iwm.get("iwm"), dict):
        iwm = iwm["iwm"]
    iwm = iwm if isinstance(iwm, dict) else {}
    organs = iwm.get("organ_trust") or {}
    frac = 0.0
    if isinstance(organs, dict) and organs:
        measured = sum(1 for v in organs.values() if isinstance(v, (int, float)) and v > 0.05)
        frac = measured / max(1, len(organs))
    return I6LiveFacts(
        world_pool_size=int(stats.get("world_pool_size") or 0),
        trace_count=int(stats.get("trace_count") or 0),
        experience_count=int(stats.get("experience_count") or 0),
        n_v_updates=int((stats.get("capability_flow") or {}).get("n_v_updates") or 0),
        organ_trust=dict(organs) if isinstance(organs, dict) else {},
        organ_measured_frac=round(frac, 4),
        self_trust=float(iwm.get("self_trust") or 0.0),
        degrade_to_baseline=bool(iwm.get("degrade_to_baseline")),
        source="arsi_checkpoint.json",
    )


def merge_facts(*parts: I6LiveFacts) -> I6LiveFacts:
    """Keep max thickness / worst flags across sources."""
    out = I6LiveFacts()
    for p in parts:
        if p is None:
            continue
        out.world_pool_size = max(out.world_pool_size, p.world_pool_size)
        out.trace_count = max(out.trace_count, p.trace_count)
        out.experience_count = max(out.experience_count, p.experience_count)
        out.manifest_cycles = max(out.manifest_cycles, p.manifest_cycles)
        out.n_v_updates = max(out.n_v_updates, p.n_v_updates)
        out.layer1_holdout = max(out.layer1_holdout, p.layer1_holdout)
        out.health_rows = max(out.health_rows, p.health_rows)
        out.calibration_n = max(out.calibration_n, p.calibration_n)
        out.organ_measured_frac = max(out.organ_measured_frac, p.organ_measured_frac)
        out.host_success_rate = max(out.host_success_rate, p.host_success_rate)
        out.paired_promote = out.paired_promote or p.paired_promote
        out.live_regression = out.live_regression or p.live_regression
        out.degrade_to_baseline = out.degrade_to_baseline or p.degrade_to_baseline
        if p.organ_trust:
            out.organ_trust = {**out.organ_trust, **p.organ_trust}
        if p.q_pass:
            out.q_pass = p.q_pass
            out.q_gate_ok = p.q_gate_ok
            out.q_failed = list(p.q_failed or [])
        if p.suite_verified:
            out.suite_verified = p.suite_verified
        if p.self_trust:
            out.self_trust = p.self_trust
        if p.source:
            out.source = f"{out.source}|{p.source}" if out.source else p.source
    return out
